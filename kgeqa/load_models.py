import pandas as pd
import pymagnitude

from .params import (
    WORD_VECTORS_MODEL_LOCATION,
    ENTITY_VECTORS_DB_LOCATION,
    RELATION_VECTORS_DB_LOCATION,
    KG_DATASET_LOCATION,
)
from .utils.logger import logging


# --- TO LOAD AT SYSTEM INIT --- #


def load_embedding_model(path: str) -> pymagnitude.Magnitude:
    # github.com/plasticityai/magnitude
    logging.info(f"loading embedding model from:\n {path}")
    vectors = pymagnitude.Magnitude(path=path)
    return vectors


def read_kg_data():
    logging.info(f"reading knowledge graph data from:\n{KG_DATASET_LOCATION}")
    df = pd.read_csv(KG_DATASET_LOCATION)
    # lower case heads/tails
    df["h"] = df["h"].apply(lambda x: x.lower())
    df["t"] = df["t"].apply(lambda x: x.lower())
    return df


# -- Word Embeddings Model
EMBEDDING_MODEL = load_embedding_model(WORD_VECTORS_MODEL_LOCATION)
# -- Entity Embeddings Model
ENTITY_VECTORS_MODEL = load_embedding_model(ENTITY_VECTORS_DB_LOCATION)
ENTITY_VECTORS_KEYS = set(map(lambda x: x[0].lower(), ENTITY_VECTORS_MODEL))
# -- Relation Embeddings Model
RELATION_VECTORS_MODEL = load_embedding_model(path=RELATION_VECTORS_DB_LOCATION)
RELATION_VECTORS_KEYS = set(map(lambda x: x[0].lower(), RELATION_VECTORS_MODEL))
# -- Knowledge (Facts) Graph dataset
KG_DATABASE = read_kg_data()
