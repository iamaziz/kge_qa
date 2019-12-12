"""
The Main KGE_QA Algorithm
"""
# Wed Oct 30 21:33:30 EDT 2019
# Author: Aziz Altowayan
from typing import List, Tuple, Iterator

from .config import Token
from .utils.logger import logging
from .params import TYPE_ENTITY, TYPE_RELATION
from .preprocess import tokenize
from .model import (
    generate_embedding,
    is_true_key,
    decide_closest_neighbor,
    form_pairs,
    pick_potential_pair,
    find_closest_pair_in_kg,
)

########################
# -- Main Algorithm -- #
########################


class KGE_QA(object):
    """Factoid Question Answering based on Knowledge Graph Embeddings"""

    input_tokens: Iterator[Token]
    labeled_tokens: List[Token]
    swapped_tokens: List[Token]
    candidate_pairs: List[Tuple[Token, Token]]
    results: Tuple[str, str, List[str]]

    def tokenize(self, question: str) -> None:
        self.input_tokens = tokenize(question)

    def _phase1_identify_and_label_tokens(self):
        """label each token i.e. POS either ENT, REL, or OTHER"""

        self.labeled_tokens = list()
        for token in self.input_tokens:

            # if the token is a true ENT/REL, keep as is and skip finding its neighbors
            if is_true_key(token):
                self.labeled_tokens.append(token)
                continue

            # -- Embed the input `token.name` into a vector (and update `token.vector` value)
            token.vector = generate_embedding(token)

            # -- The important magic happens here:
            # decide the type of the input token by matching
            # it with its closest neighbor in our ENT/REL models
            closest_neighbor, distance = decide_closest_neighbor(
                current_token=token, processed_tokens=self.labeled_tokens
            )

            # -- Fill in current_token's params
            token.type = closest_neighbor.type
            token.type_confidence = distance
            token.closest_token = closest_neighbor

            # -- Add token to the processed tokens
            self.labeled_tokens.append(token)

    def _phase2_swap_from_input_to_kg_tokens(self):
        v = [(x.name, x.type) for x in self.labeled_tokens]
        logging.info(f"\nLABELED TOKENS:\n{v}")
        self.swapped_tokens = [t.closest_token for t in self.labeled_tokens]

        v = [(x.name, s.name) for x, s in zip(self.labeled_tokens, self.swapped_tokens)]
        logging.info(f"\nSWAPPED TOKENS:\n{v}")

    def _phase3_form_incomplete_triplets(self):
        # -- keep only ENT/REL types i.e. discard the type OTHER
        ents_kge_tokens = [t for t in self.swapped_tokens if t.type == TYPE_ENTITY]
        rels_kge_tokens = [t for t in self.swapped_tokens if t.type == TYPE_RELATION]
        # -- form incomplete triplets pairs
        self.candidate_pairs = form_pairs(ents_kge_tokens, rels_kge_tokens)

    def _phase4_find_missing_tail(self):
        """receives incomplete pairs, pick one, and complete it with its missing tail(s)"""
        if not self.candidate_pairs:
            self.results = None, None, "Invalid question!"
            return
        head, relation = pick_potential_pair(self.candidate_pairs)
        tail = find_closest_pair_in_kg(head, relation)
        self.results = head, relation, tail


def answer(question: str) -> Tuple[str, str, List[str]]:
    """The pipeline for the answering task"""

    kgeqa = KGE_QA()
    kgeqa.tokenize(question)

    # -- phase 1: extract entities and relations from the input question
    kgeqa._phase1_identify_and_label_tokens()

    # -- phase 2: swap from the question's tokens to the closest matched tokens from our models
    # TODO: perhaps here we need to keep one entity and one relation only!
    kgeqa._phase2_swap_from_input_to_kg_tokens()

    # -- phase 3: here we form ENT/REL pairs from the parsed input token (the incomplete triplet)
    # and pick the most likely one
    kgeqa._phase3_form_incomplete_triplets()

    # -- phase 4: (fact-completion) get the missing entity (of the incomplete triplet) from the KG dataset
    kgeqa._phase4_find_missing_tail()

    entity, relation, result = kgeqa.results
    print(f"Answer: {result}")

    return entity, relation, result


############################
# -- Command-line Entry -- #
############################


def main():
    question = input("enter your question >> ")
    # print(f"question: {question}")
    ans = answer(question)
    print(f"KG pair: {ans[0]}->{ans[1]}->?")
    # print(f"answer: {ans}")


if __name__ == "__main__":
    try:
        while True:
            main()
    except KeyboardInterrupt:
        print("\nEnd of QA system! Good bye.")
