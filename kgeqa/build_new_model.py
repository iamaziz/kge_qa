"""
Build embedding models for new domain knowledge from a knowledge graph dataset (csv file)

Usage:

    # from CLI
    $ python build_new_model.py -kg_dataset <FULL-PATH-TO-CSV-FILE>

    # OR from module-level
    from kge_qa.kgeqa.build_new_model import BuildKGModels
    BuildKGModels().run("<full-path-to-csv-file>")

INPUT:
    csv file contains knowledge graph dataset with three columns h,r,t
OUTPUT:
   Two new embedding models, one for entities and one for relations
"""
from typing import List
from pymagnitude import Magnitude

import pandas as pd


class BuildKGModels:
    vector_model: Magnitude
    csv_file: str
    entities: List[str]
    relations: List[str]
    ENT_VEC_OUTPUT: str = "data/ENT.vec"
    REL_VEC_OUTPUT: str = "data/REL.vec"

    def __init__(self):
        from .load_models import EMBEDDING_MODEL

        self.vector_model = EMBEDDING_MODEL

    def run(self, csv_file):
        self.csv_file = csv_file
        self.entities, self.relations = self.read_kg_data(self.csv_file)
        self.build_vectors_file(self.entities, self.ENT_VEC_OUTPUT)
        self.build_vectors_file(self.relations, self.REL_VEC_OUTPUT)
        self._convert_to_magnitude_format()

    @staticmethod
    def read_kg_data(csv_file):
        """reads csv as dataframe and returns entities and relations"""
        print(f"Started a model builder for data from: {csv_file}")
        df = pd.read_csv(csv_file)
        df.columns = ["h", "r", "t"]
        entities = list(set(df["h"].tolist() + df["t"].tolist()))
        relations = list(set(df["r"].tolist()))
        return entities, relations

    def build_vectors_file(self, word_tokens: List[str], out_file: str):
        """builds MODEL.vec"""
        print(f"Building a new embedding model for {len(word_tokens)} tokens ..")
        with open(out_file, "w") as out:
            out.write(f"{len(word_tokens)} {self.vector_model.dim}\n")
            for e in word_tokens:
                v = self.vector_model.query(e)  # Magnitude.query()
                str_vec = " ".join(map(str, v))  # ndarray to str
                line = f"{e} {str_vec}\n"
                out.write(line)
        print(f"Done. See output: {out_file}")

    def _convert_to_magnitude_format(self):
        """converts MODEL.vec to MODEL.vec.magnitude"""
        import os

        print(f"Converting models to .magnitude format ..")
        cmd1 = f"python -m pymagnitude.converter -i {self.ENT_VEC_OUTPUT} -o {self.ENT_VEC_OUTPUT}.magnitude"
        cmd2 = f"python -m pymagnitude.converter -i {self.REL_VEC_OUTPUT} -o {self.REL_VEC_OUTPUT}.magnitude"
        os.system(cmd1)
        os.system(cmd2)
        print(f"Done")

        # -- override the previous default dataset
        cmd1 = f"cp {self.csv_file} data/KG.csv"
        os.system(cmd1)


def main(input_file):
    """create ENT.vec and REL.vec models from `input_file`"""
    builder = BuildKGModels()
    builder.run(input_file)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-csv", "--kg_dataset", default="data/KG.csv")
    args = parser.parse_args()
    csv_file_path = args.kg_dataset

    main(csv_file_path)
