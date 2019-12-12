"""The Magic"""
# Author: Aziz Altowayan (November, 2019)
import itertools
from typing import List, Tuple


from .config import Token, Tensor
from .utils.logger import logging
from .params import (
    TYPE_RELATION,
    TYPE_ENTITY,
    TYPE_OTHER,
    _THRESHOLD_MAX_CONFIDENCE,
    _THRESHOLD_MIN_CONFIDENCE,
)
from .load_models import (
    EMBEDDING_MODEL,
    ENTITY_VECTORS_MODEL,
    ENTITY_VECTORS_KEYS,
    RELATION_VECTORS_MODEL,
    RELATION_VECTORS_KEYS,
    KG_DATABASE,
)


def generate_embedding(token: Token) -> Tensor:
    """Use a pre-trained model to get/calculate a vector for the input `token.name`"""
    logging.info(f"\nGenerating embedding for current TOKEN: '{token.name}'")
    tensor = EMBEDDING_MODEL.query(token.name)  # return: ndarray i.e. Token
    return tensor


def find_closest(word: Token, model, top_n=3) -> List[Tuple[str, float]]:
    """
    returns the `topn` closest words to `word` from `model`
    return structure [(word1, cosine_dist), (word2, cosine_dist), (...)]
    """
    return model.most_similar(word.name, topn=top_n)


def find_closest_relations(token: Token, n: int = 3) -> List[Tuple[Token, float]]:
    """Computes distances between `token` and relations in our `REL.vec`"""
    closest_n_relations = find_closest(
        word=token, model=RELATION_VECTORS_MODEL, top_n=n
    )
    results = []
    for closest, distance in closest_n_relations:
        neighbor = Token(name=closest, type=TYPE_RELATION, type_confidence=1.0)
        neighbor.vector = None  # TODO: consider adding the neighbor's vector as well.
        results.append((neighbor, distance))
    m = f"Closest '{n}' neighbors of type 'RELATION': {closest_n_relations}"
    logging.info(m)
    return results


def find_closest_entities(token: Token, n: int = 3) -> List[Tuple[Token, float]]:
    """Computes distances between `token` and all entities in our `ENT.vec`"""
    closest_n_entities = find_closest(word=token, model=ENTITY_VECTORS_MODEL, top_n=n)
    results = []
    for closest, distance in closest_n_entities:
        neighbor = Token(name=closest, type=TYPE_ENTITY, type_confidence=1.0)
        neighbor.vector = None  # TODO: consider add the neighbor's vector as well.
        results.append((neighbor, distance))
    m = f"Closest '{n}' neighbors of type 'ENTITY': {closest_n_entities}"
    logging.info(m)
    return results


def is_true_key(token: Token) -> bool:
    """check if `token.name` is already a true entity/relation key in our model"""
    if token.name in ENTITY_VECTORS_KEYS:
        m = f"The token '{token.name}' is a True ENTITY in 'ENTITY_VECTORS_KEYS'"
        logging.info(m)
        token.type = TYPE_ENTITY
        token.type_confidence = 1.0
        token.closest_token = token  # closest token is self
        return True
    if token.name in RELATION_VECTORS_KEYS:
        m = f"The token '{token.name}' is a True RELATION in 'RELATION_VECTORS_KEYS'"
        logging.info(m)
        token.type = TYPE_RELATION
        token.type_confidence = 1.0
        token.closest_token = token
        return True
    return False


def already_found_entity_token(tokens: List[Token]) -> bool:
    for token in tokens:
        if token.type == TYPE_ENTITY and token.type_confidence > 0.4:
            return True
    return False


def sequence_matcher(target: str, word: str) -> float:
    from difflib import SequenceMatcher as SM

    ratio = SM(a=target.lower(), b=word.lower()).ratio()
    logging.info(f".. SM({target}, {word}) = {ratio:.2f}")
    return ratio


def decide_closest_neighbor(
    current_token: Token, processed_tokens: List[Token]
) -> Tuple[Token, float]:
    """
    Input: the currently being processed `token` and previously `processed_tokens`
    Return: the most likely closest neighbor to `token` along the with distance
    """

    # -- get token's nearest neighbors in our embeddings models,
    # neighbors are mixture of ENTs and RELs types
    closest_entities = find_closest_entities(current_token, n=3)
    closest_relations = find_closest_relations(current_token, n=3)
    neighbors = closest_entities + closest_relations

    # -- sort neighbors by CosineSim (in DESC order),
    # and take the one with max cosine similarity score
    neighbors.sort(key=lambda tup: tup[1], reverse=True)
    neighbor_token, distance = neighbors[0]  # max(cosine similarity of the neighbors)

    if distance < _THRESHOLD_MAX_CONFIDENCE:
        # -- take the max of (sequenceMatcher ratio + cosine similarity) / 2
        candidates = {
            t.name: round((c + sequence_matcher(current_token.name, t.name)) / 2, 2)
            for t, c in neighbors
        }
        # -- sort by the avg score (in DESC order) and take the max
        candidates = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)
        closest, distance = candidates[0]
        neighbor_token, distance = (
            [t for t, c in neighbors if t.name == closest][0],
            distance,
        )
        logging.info(f"candidates avg. distance: {candidates}")

    m = f"The closest neighbor: '{neighbor_token.name}', type: '{neighbor_token.type}', distance: {distance:.2f}"
    logging.info(m)

    if distance < _THRESHOLD_MIN_CONFIDENCE:
        # if the input token is neither ENT nor REL
        neighbor_token.name = TYPE_OTHER
        neighbor_token.type = TYPE_OTHER
        distance = 0.0

    return neighbor_token, distance  # (Token, cosine_similarity)


def form_pairs(
    entities: List[Token], relations: List[Token]
) -> List[Tuple[Token, Token]]:
    """formulate `(entity, relation)` from the inputs"""
    if len(entities) < 1:
        ERROR_MSG = "INVALID! no 'Entities' found in the question"
        logging.error(ERROR_MSG)
    if len(relations) < 1:
        ERROR_MSG = "INVALID! no 'Relations' found in the question"
        logging.error(ERROR_MSG)
    product = list(itertools.product(entities, relations))
    logging.info(f"\nCandidate pairs: {[(p1.name, p2.name) for p1, p2 in product]}")
    return product


def pair_in_df_head_relation(h: str, r: str):
    """returns rows where h->r occurs"""
    row = KG_DATABASE[(KG_DATABASE["h"] == h) & (KG_DATABASE["r"] == r)]
    return row


def pick_potential_pair(pairs: List[Tuple[Token, Token]]) -> Tuple[str, str]:
    """
    takes a list of possible ent-rel pairs, and
    returns the most probable pair
    by e.g. closeness to a pair in our KGE, or match to a pair in our KG dataset
    """
    # Assumes input pairs always: [(ENT, REL), (ENT, REL), ... ]
    pair = pairs[0]  # TODO: find a better way to pick the true triplets (see below)jj
    # -- swap to a pair exists in our model
    # check if the pair is: HEAD->RELATION
    for ent, rel in pairs:
        db_pair = pair_in_df_head_relation(ent.name, rel.name)
        if not db_pair.empty:
            pair = ent, rel
            answer = db_pair["t"].iloc[0]

    ent, rel = pair  # assume
    logging.info(f"\nSelected pair: ({ent.name}, {rel.name})")
    return ent.name, rel.name


def find_closest_pair_in_kg(entity: str, relation: str) -> List[str]:
    """
    use kge to find closest triplet to `input_pair`
    NOTE: perhaps no need to use KGE just use our original FB15K dataset to match the correct pair
    """
    df = KG_DATABASE  # KGE dataset
    t = df["t"][(df["h"] == entity) & (df["r"] == relation)]
    if not t.empty:
        return t.values
    return ["No answer found!"]
