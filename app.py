"""UI for KGE QA

To start:
    $ streamlit run app.py

see streamlit docs: https://streamlit.io/docs/getting_started.html#set-up-your-virtual-environment


Author: Aziz Altowayan (Nov, 2019)
"""

import sys

import streamlit as st

# sys.path.append("/Users/Aziz/Dropbox/thesis/code/")
sys.path.append("../")

from kge_qa.kgeqa.main import answer
from kge_qa.kgeqa.params import HOWTO_VIDEO_PATH
from kge_qa.kgeqa.model import (
    ENTITY_VECTORS_MODEL,
    RELATION_VECTORS_MODEL,
    ENTITY_VECTORS_KEYS,
    RELATION_VECTORS_KEYS,
    KG_DATABASE,
)
from kge_qa.kgeqa.utils.logger import logging


def render_intro():
    st.title("KGE QA System")
    s = "**Factoid-Question Answering System Based on Knowledge Graph Embeddings**"
    st.markdown(s)
    help_ = st.checkbox("Show help")
    if help_:
        st.write(
            "This system answers simple factual questions such as `who acted in titanic movie?` "
            "or `what's the spoken language in Indonesia?`. "
            "The goal is to demonstrate how to leverage Knowledge Graphs Embeddings to build such system."
        )
        st.markdown("<hr>")
        st.info(
            "**What kind of questions can be asked?**<br/>"
            "This system supports a specific domain knowledge.<br/>"
            "To make sure the asked question falls under the covered domain knowledge, "
            "a question should contain _at least_ **one entity (head)** and **one relation** from the domain knowledge facts.<br/>"
            "See '**Explore the domain knowledge**' on the left sidebar to learn more about the supported facts and their ENTities/RELations.<br/>"
            "<hr>"
            "_NOTICE_: A key characteristic of the system, however, is that it is possible to "
            "ask about the same piece of information in a variety of (syntactic/semantic) linguistic forms.<br/>"
            "In other words, the same question might be asked in different forms e.g. <br/>"
            "- `Troy movie is written by whom?`</br>"
            "- `who wrote the movie troy?`"
        )
        _how_to_use_video()


def st_draw_domain_ontology():
    """draw the figure of hte domain ontology
    add the ".png" file
    see: https://streamlit.io/docs/api.html#streamlit.graphviz_chart
    """
    view_domain = st.sidebar.checkbox("View domain ontology")
    if not view_domain:
        return
    st.markdown("<center>QUESTIONS DOMAIN ONTOLOGY</center>")
    st.graphviz_chart(
        """
        digraph {
             rankdir=LR; // Left to Right, instead of Top to Bottom
             film -> director[label="directed by"]
             film -> author[label="written by"]
             director -> film[label="directed"]
             actor -> film[label="played in"]
             language -> language[label="influence"]
             city -> country[label="located_in"]
        }   
        """
    )
    if st.checkbox("See the complete ontology of FB15K"):
        st.image(
            "img/fb15__ontology.png",
            caption="Complete ontology of the filtered FB15K Knowledge Graph",
            use_column_width=True,
        )


def st_draw_answer_triplet(h, r, t):
    """return graphviz `dot langauge` format diagram
    see: https://graphs.grevian.org/example
    """
    temp = """
        digraph {
             rankdir=LR; // Left to Right, instead of Top to Bottom 
             HEAD -> TAIL[label="RELATION",color=green] 
        }
        """
    temp = temp.replace("HEAD", f"{h}")
    temp = temp.replace("TAIL", f"{t}")
    temp = temp.replace("RELATION", f"{r}")
    print(temp)
    st.graphviz_chart(temp)


def st_render_answer(results):
    head, relation, result = results  # str, str, List[str]
    if not head:  # case invalid question (could not form pairs)
        st.error("Invalid question!")
        return
    st.text(f"Answer:")
    st.write(result)
    clean_result = [
        x.replace(" ", "_").replace(".", "_").replace("-", "_") for x in result
    ]
    joined_tails = ",".join([x for x in clean_result])
    dot_edge_entry = (head.replace(" ", "_"), relation.replace(" ", "_"), joined_tails)
    st_draw_answer_triplet(*dot_edge_entry)
    # TODO: flush the diagram of the previous answer, if no new answer found


def _sb_entities_model():
    if st.sidebar.checkbox("ENT model"):
        w = st.sidebar.text_input("Type a word to find its closest entities:")
        if w:
            # -- display dataframe
            result = ENTITY_VECTORS_MODEL.most_similar(w)
            st.sidebar.dataframe(result)

            # -- show visualization
            if st.sidebar.button("Show plot"):
                tsne_plot_most_similar(w, ENTITY_VECTORS_MODEL, 30, "ENTITIES")


def _sb_relations_model():
    if st.sidebar.checkbox("REL model"):
        w = st.sidebar.text_input("Type a word to find its closest relations:")
        if w:
            # -- display dataframe
            result = RELATION_VECTORS_MODEL.most_similar(w)
            st.sidebar.dataframe(result)

            # -- show visualization
            if st.sidebar.button("Show plot"):
                tsne_plot_most_similar(w, RELATION_VECTORS_MODEL, 20, "RELATIONS")


def _sb_display_supported_ents_rels():
    if not st.sidebar.checkbox("View ENT/REL keys"):
        return
    n = len(ENTITY_VECTORS_KEYS)
    if st.sidebar.button(f"{n} entities"):
        st.sidebar.dataframe(ENTITY_VECTORS_KEYS)

    n = len(RELATION_VECTORS_KEYS)
    if st.sidebar.button(f"{n} relations"):
        st.sidebar.dataframe(RELATION_VECTORS_KEYS)


def _sb_display_kg_facts():
    if not st.sidebar.checkbox("View supported facts"):
        return
    df = KG_DATABASE  # read_kg_dataset()

    # -- render dataframe
    st.markdown("<hr>")

    st.write(f"> Number of facts: {KG_DATABASE.shape[0]}")
    term = st.text_input("Search for a fact:")
    if term:
        # search `term` in df
        st.dataframe(
            df[
                df["h"].str.match(term)
                | df["r"].str.match(term)
                | df["t"].str.match(term)
            ]
        )
    else:
        st.dataframe(KG_DATABASE)


def _sb_display_kg_facts_and_questions_ontology():
    st_draw_domain_ontology()
    _sb_display_kg_facts()


def _sb_supplementals_play_with_models():
    st.sidebar.markdown("**Explore the domain knowledge**:")
    _sb_display_kg_facts_and_questions_ontology()
    _sb_display_supported_ents_rels()
    st.sidebar.markdown("**Explore embedding models**:")
    _sb_entities_model()
    _sb_relations_model()


def _sb_signature():
    st.sidebar.markdown("<hr>")
    if st.sidebar.button("About"):
        st.sidebar.markdown("<sub>Author: Aziz Altowayan</sub>")
        st.sidebar.markdown("<sup>aa10212w@pace.edu</sup>")
        st.sidebar.markdown("<sup>November, 2019</sup>")


def _sb_build_new_kge_models():
    """build a new domain knowledge models from a KG csv file"""

    st.sidebar.markdown("**Build your own domain**:")
    if not st.sidebar.checkbox("Create new knowledge domain"):
        return

    st.markdown("<hr>")
    st.markdown("# Building a new knowledge domain")
    # -- read local file
    filename = st.text_input(
        "Enter the complete .csv file path of your knowledge graph "
        "(FORMAT: three columns h r t)"
    )
    if not filename:
        return
    try:
        with open(filename) as in_file:
            st.text(f"Creating a new domain knowledge from:\n\t{filename}")

        # st.text(f"Number of facts: {}") # TODO: count the csv line

        # -- run model builder
        st.text("Building KGE embedding models ... ")
        from kge_qa.kgeqa.build_new_model import BuildKGModels

        builder = BuildKGModels()
        builder.run(filename)
        st.text(
            "Done. Created new models:\n\t"
            "data/ENT.vec, data/ENT.vec.magnitude, data/REL.vec, data/REL.vec.magnitude"
        )
        st.info("Reboot KGE QA system to load the new domain")

    except FileNotFoundError:
        st.error("File not found.")


def tsne_plot_most_similar(target, model, topn=10, type_="Entities"):
    """visualize the embeddings of the most similar words"""
    # source: https://www.kaggle.com/jeffd23/visualizing-word-vectors-with-t-sne

    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    # -- font size (see: https://stackoverflow.com/a/45297434/2839786)
    import matplotlib as mpl

    mpl.rcParams["font.size"] = 34

    closest_tokens = [w for w, c in model.most_similar(target, topn=topn)]
    vectors = [model.query(target)]
    tokens = [target]

    for word in closest_tokens:
        vectors.append(model.query(word))
        tokens.append(word)

    tsne_model = TSNE(
        perplexity=10, n_components=2, init="pca", n_iter=2500, random_state=23
    )
    new_values = tsne_model.fit_transform(vectors)

    x = [value[0] for value in new_values]
    y = [value[1] for value in new_values]

    plt.figure(figsize=(25, 25))
    plt.title(f"Closest {topn} {type_} to '{target}'")
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(
            tokens[i],
            xy=(x[i], y[i]),
            xytext=(15, 5),
            textcoords="offset points",
            ha="right",
            va="bottom",
        )
    # with fontsize -- To prevent the xlabel being cut off
    plt.gcf().set_tight_layout(True)

    # Streamlit equivalent of "plt.show()"
    st.pyplot()


def _sb_evaluation_form():
    st.sidebar.markdown("> [Open evaluation form](https://forms.gle/ZtNABLKap8x8fnFM8)")


def _how_to_use_video():
    """display a video clip for how to ask questions - called from `render_intro()`"""
    if st.button("Click here for a short video on HowTo ask a question"):
        video_file = open(HOWTO_VIDEO_PATH, "rb")
        video_bytes = video_file.read()
        st.video(video_bytes)


#######################
# ==== START APP ==== #
def run():
    logging.info("KGE_QA APP is READY!")
    render_intro()
    question = st.text_input("Enter your question: ")
    if question:
        results = answer(question)  # return: str, str, List[str]
        st_render_answer(results)
    _sb_supplementals_play_with_models()
    _sb_build_new_kge_models()
    _sb_evaluation_form()
    _sb_signature()


if __name__ == "__main__":
    run()
