"""
Preprocess and Tokenize the input question and output.

Author: Aziz Altowayan (Nov, 2019)
"""
from typing import List, Iterator
import re

from .config import Token
from .params import STOPWORDS
from .model import ENTITY_VECTORS_KEYS, RELATION_VECTORS_KEYS
from .utils.logger import logging


def _remove_stopwords(raw_tokens: List[str]) -> List[str]:
    logging.info(f"Raw TOKENS:\n\t{raw_tokens}")
    return [t for t in raw_tokens if t not in STOPWORDS]


def _ngrams(s: List[str], n: int = 2, i: int = 0) -> Iterator[List[str]]:
    """
    For example,
    >>> s = "Who is the author of".split()
    >>> list(_ngrams(s))
    [['Who', 'is'], ['is', 'the'], ['the', 'author'], ['author', 'of']]
   """
    while len(s[i : i + n]) == n:
        yield s[i : i + n]
        i += 1


def _join_subwords(tokens: List[str]) -> List[str]:
    """This should take care of compound Entities (i.e. NER) and Relations
    How: for every 2-gram words check if they are in our embedding models' keys
    # models: ENT.vec and REL.vec
    # e.g.
    'new york' -> 'new_york'
    'tom cruise' -> 'tom_cruise'
    'directed by' -> 'directed_by'
    'author of' -> 'author_of' # TODO: 'of' was removed in stopwords (retain it)
    """
    new_tokens = []
    grams = _ngrams(tokens)
    i = 0
    while i < len(tokens):
        try:
            pair = next(grams)
        except StopIteration:
            if i < len(tokens):
                new_tokens.append(tokens[i])  # last word in tokens
            break
        joined = "_".join(pair)
        spaced = " ".join(pair)
        if joined in ENTITY_VECTORS_KEYS or joined in RELATION_VECTORS_KEYS:
            new_tokens.append(joined)
            i += 2
        elif spaced in ENTITY_VECTORS_KEYS or spaced in RELATION_VECTORS_KEYS:
            new_tokens.append(spaced)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    return new_tokens


def tokenize(text: str) -> Iterator[Token]:
    # remove symbols | split | lower case
    logging.info(f"\n\n--------\nTokenizing the input query:\n\t'{text}'")
    tokens = re.findall("[a-z1-9_]+", text.lower())

    # assert len(tokens) > 2, "INVALID question! A question should have more than 2 words"
    if len(tokens) < 3:
        logging.error("INVALID question! A question should have more than 2 words")

    # TODO: consider using an nlp package for NER extraction

    # 1) remove stop words except "by" e.g. "directed by"
    tokens = _remove_stopwords(tokens)

    # 2) consider concatenating sub-words as one token e.g. "directed by" -> "directed_by"
    tokens = _join_subwords(tokens)

    logging.info(f"Filtered TOKENS:\n\t{tokens}")
    return map(Token, tokens)
