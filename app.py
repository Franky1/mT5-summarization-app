"""
Python app to demo mT5/XLSum.
"""
import json

import streamlit as st
import pandas as pd

from summarize import mT5, XLSUM_LANGS

def init():
    """Loads the mT5 model from HF.
    """
    if "model" not in st.session_state:
        st.session_state.model = mT5()
    if "canned" not in st.session_state:
        with open('canned_text.json','r', encoding='utf-8') as fp:
            st.session_state.canned = json.load(fp)

# layout ----------------
# on session start
st.set_page_config("NLP Summarization Demo", page_icon=":book:", layout="wide")
with st.spinner("Loading Model..."):
    init()

# sidebar
input_mode = st.sidebar.radio("Input Mode", ["Canned Text", "Text Box"])

# title and desc
st.title("NLP Multi-Lingual Summarization App")
st.markdown(
    "This app allows you to test cutting edge news summarization models on \
    text that is in-domain like news from CNN or BBC Korea; and out-of-domain \
    (OOD) such as the Korean Journal Of Medicine and ArXiv."
)

# input form area
st.header("Input Area")
input_form = st.empty()

# output area
col1, col2 = st.columns(2)
with col1:
    st.header("Full Input")
    full_input = st.empty()

with col2:
    st.header("Summarization")
    full_output = st.empty()

# about section
with st.expander("About the Model"):
    st.markdown(
        """This model is [mT5](https://arxiv.org/abs/2010.11934), trained on the multilingual dataset [XLSum](https://aclanthology.org/2021.findings-acl.413/) by the [BUET CSE](https://cse.buet.ac.bd/research/index.php) NLP Group. The model can be found on the [:hugging_face: model repository](https://huggingface.co/csebuetnlp/mT5_multilingual_XLSum). It was intended to summarize news articles, research article summarization is out of scope."""
    )
    st.markdown("The following languages were used to train the mT5 Model:")
    st.dataframe(
        pd.DataFrame(data=XLSUM_LANGS, columns=["Language", "Number of XLSum Articles"])
    )

# interactivity -----------------

# set the input form depending on the sidebar input mode
if input_mode == "Canned Text":
    with input_form.form("Input"):
        text_key = st.selectbox("Select a canned input", options=st.session_state.canned.keys())
        submitted = st.form_submit_button("Summarize!")
else:  # if input_mode == "Text Box":
    with input_form.form("Input"):
        text_area = st.text_area("Your Text")
        submitted = st.form_submit_button("Summarize!")

# run the model on submit
if submitted:
    # collect the input fields
    if input_mode == "Canned Text":
        text_of_input = st.session_state.canned[text_key]
    else:  # if input_mode == "Text Box":
        text_of_input = text_area
    
    full_input.markdown(text_of_input)

    # pass input to model and print output
    with st.spinner("Summarizing..."):
        text_of_output = st.session_state.model.run(text_of_input)
        full_output.markdown(text_of_output)
