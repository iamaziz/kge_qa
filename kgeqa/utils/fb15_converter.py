"""from entity id to entity name or definition (FB15K dataset)"""
from qa_system.utils.logger import logging

logging.info("loaded FB15 dictionary.")
# also: http://webpage.pace.edu/aa10212w/thesis/data/fb15/entityWords.txt
ENTITY_DESC_FILE = "/Users/Aziz/Downloads/rel-embedding/RelatedWork/data/data_descriptions/entity_word/entityWords.txt"


def __read_entity_descriptions():
    """
    read the reference file of the entity descriptions file and load as a pandas dataframe
    :return: pd.DataFrame
    """
    import pandas as pd

    df_ = pd.read_csv(ENTITY_DESC_FILE, names=["mid", "id", "definition"], sep="\t")
    del df_["id"]
    df_.set_index("mid", inplace=True)
    return df_


df = __read_entity_descriptions()
FB15_DESC_DICT = df["definition"].to_dict()


def mid2description(mid: str = "/m/027rn"):
    """returns FreeBase description of the entity `mid`
    descriptions based on: entityWords.txt from https://github.com/xrb92/DKRL
    """
    try:
        return FB15_DESC_DICT[mid]
    except KeyError:
        return f"UNKNOWN: {mid}"


def mid2concepts(
    input_csv="../../../data/all_domains_ids.csv", output_csv="../data/all_FB15K_converted.csv"
):
    # see: ..thesis/code/data/ontology_description/mid2concepts.ipynb
    # source of mapping data here: https://github.com/brmson/dataset-factoid-webquestions/tree/master/d-freebase-mids

    # -- read the mapping data from the source
    mapping1 = "https://raw.githubusercontent.com/brmson/dataset-factoid-webquestions/master/d-freebase-mids/trainmodel.json"
    mapping2 = "https://raw.githubusercontent.com/brmson/dataset-factoid-webquestions/master/d-freebase-mids/val.json"
    mapping3 = "https://raw.githubusercontent.com/brmson/dataset-factoid-webquestions/master/d-freebase-mids/test.json"
    mapping4 = "https://raw.githubusercontent.com/brmson/dataset-factoid-webquestions/master/d-freebase-mids/devtest.json"

    import requests

    data1 = requests.get(mapping1).json()
    data2 = requests.get(mapping2).json()
    data3 = requests.get(mapping3).json()
    data4 = requests.get(mapping4).json()

    def get_map(data):
        flattened = {}
        for d in data:
            concepts_mids = d["freebaseMids"]
            for c in concepts_mids:
                concept = c["concept"]
                mid = c["mid"]
                mid = f"/{mid.replace('.', '/')}"
                flattened[mid] = concept
        return flattened

    flat1 = get_map(data1)
    flat2 = get_map(data2)
    flat3 = get_map(data3)
    flat4 = get_map(data4)
    flattened = {**flat1, **flat2, **flat3, **flat4}

    # -- read the `mid` filtered FB15K dataset
    import pandas as pd

    fb15k = pd.read_csv(input_csv)

    # -- convert `mid` to actual concepts
    fb15k["h"] = df["h"].apply(lambda x: flattened[x] if x in flattened else x)
    fb15k["t"] = df["t"].apply(lambda x: flattened[x] if x in flattened else x)
    # -- keep only triples with converted heads/tails concepts
    cleaned = fb15k[
        (~fb15k["h"].str.startswith("/")) & (~fb15k["t"].str.startswith("/"))
    ]
    concepts = cleaned["h"].tolist() + cleaned["t"].tolist()
    # -- remove paths from relations
    cleaned = cleaned["r"] = cleaned["r"].apply(lambda x: str(x.split("/")[-1]))
    cleaned.to_csv(output_csv, index=False)
