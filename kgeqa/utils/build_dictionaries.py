"""
Create dictionary for Entities and Relations
from KG based on the filtered FB15K dataset
"""
# -- building ENT_DICT.json
"""
consider:
- words variations
- data structure

{
"mid":  
}
"""
# -- building REL_DICT.json
"""
things to consider:
- relation variations
- data structure
"""

from typing import List
from pymagnitude import Magnitude


def load_model():
    p = "/Users/Aziz/.magnitude/word2vec_heavy_GoogleNews-vectors-negative300.magnitude"
    vectors = Magnitude(path=p)
    return vectors


model = load_model()


def get_words(KG):
    # -- Toy dataset
    # Assuming the following KGs:
    nodes = list(map(lambda x: x.split(","), KG))
    print(f"triplets: {nodes}")
    entities = list(map(lambda n: n[0], nodes)) + list(map(lambda n: n[2], nodes))
    entities = [e.strip().lower() for e in entities]  # clean
    print(f"entities: {entities}")
    relations = list(map(lambda n: n[1], nodes))
    relations = [r.strip().lower() for r in relations]  # clean
    print(f"relations: {relations}")

    return entities, relations


def build_vectors_file(vec_model: Magnitude, words: List[str], out_file="ENT.vec"):
    print(f"Building {out_file}")
    with open(out_file, "w") as out:
        out.write(f"{len(words)} {vec_model.dim}\n")
        for e in words:
            v = vec_model.query(e)
            str_vec = " ".join(map(str, v))  # ndarray to str
            line = f"{e} {str_vec}\n"
            out.write(line)


if __name__ == "__main__":

    # create toy ENT.vec and REL.vec models
    KG = [
        "Titanic, directed_by, James Cameron",
        "Tom Cruise, played_in, Top_Gun",
        "James Cameron, director, Avatar",
        "Top_Gun, cast, Tom_cruise",
        "Tom_cruise, played_in, top_gun",
        "Lisp, influenced, Python",
    ]
    ENTS, RELS = get_words(KG)
    build_vectors_file(model, words=ENTS, out_file="ENT.vec")
    build_vectors_file(model, words=RELS, out_file="REL.vec")
