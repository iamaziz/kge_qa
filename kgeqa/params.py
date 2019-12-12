from pathlib import Path
from os.path import dirname, abspath

HOME_DIR = str(Path.home())
PARENT_DIR = dirname(dirname(abspath(__file__)))


# WORD_VECTORS_MODEL_LOCATION = f"{HOME_DIR}/.magnitude/word2vec_heavy_GoogleNews-vectors-negative300.magnitude"
# $ curl http://magnitude.plasticity.ai/word2vec/medium/GoogleNews-vectors-negative300.magnitude -o GoogleNews-vectors-negative300.magnitude
# $ curl http://magnitude.plasticity.ai/fasttext/medium/wiki-news-300d-1M-subword.magnitude -o ~/.magnitude/ft-wiki-news-300d-1M-subword.magnitude
WORD_VECTORS_MODEL_LOCATION = f"{HOME_DIR}/.magnitude/ft-wiki-news-300d-1M-subword.magnitude"

RELATION_VECTORS_DB_LOCATION = f"{PARENT_DIR}/data/REL.vec.magnitude"  # path to REL_DICT.vec
ENTITY_VECTORS_DB_LOCATION = f"{PARENT_DIR}/data/ENT.vec.magnitude"  # path to ENT_DICT.vec
KG_DATASET_LOCATION = f"{PARENT_DIR}/data/KG.csv"


# -- misc
HOWTO_VIDEO_PATH = f"{PARENT_DIR}/misc/kgeqa_howto_2019-12-04_at_17.25.52.mov"

# -- the string value to use in token.type
TYPE_ENTITY = "<ENTITY>"
TYPE_RELATION = "<RELATION>"
TYPE_OTHER = "<OTHER>"

_THRESHOLD_MAX_CONFIDENCE = 0.9
_THRESHOLD_MIN_CONFIDENCE = 0.2

# -- nltk.corpus.stopwords - "by"
STOPWORDS = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "you're", "you've", "you'll", "you'd",
             "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "she's", "her", "hers",
             "herself", "it", "it's", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which",
             "who", "whom", "this", "that", "that'll", "these", "those", "am", "is", "are", "was", "were", "be", "been",
             "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but",
             "if", "or", "because", "as", "until", "while", "of", "at", "for", "with", "about", "against", "between",
             "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out",
             "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why",
             "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
             "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "don't",
             "should", "should've", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "aren't", "couldn",
             "couldn't", "didn", "didn't", "doesn", "doesn't", "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't",
             "isn", "isn't", "ma", "mightn", "mightn't", "mustn", "mustn't", "needn", "needn't", "shan", "shan't",
             "shouldn", "shouldn't", "wasn", "wasn't", "weren", "weren't", "won", "won't", "wouldn", "wouldn't"}
