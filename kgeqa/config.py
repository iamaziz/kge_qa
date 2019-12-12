# Author: Aziz Altowayan (Nov, 2019)
from dataclasses import dataclass

from numpy import ndarray as Tensor


@dataclass  # python3.7+
class Token:
    name: str
    vector: Tensor = float("NaN")
    type: str = ""
    type_confidence: float = float("NaN")
    closest_token = None  # type: "Token"


@dataclass
class Node:
    # a Triplet is (Node1.token.type.ent, Node2.token.type.relation, Node3.token.type.ent)
    name: str
    token: Token


@dataclass
class Triplet:
    name: str
    e1: Token  # = None
    e2: Token  # = None
    r: Token  # = None
