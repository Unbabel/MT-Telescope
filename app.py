import streamlit as st
import pandas as pd
import numpy as np
from comet.models import download_model

import plotly.figure_factory as ff
from PIL import Image
import altair as alt
from metrics.bleu import BLEU
from metrics.chrf import chrF
from metrics.comet import COMET
from metrics.length_ratio import LengthRatio
from testset import PairedTestset, read_lines
from metrics.result import PairedResult

# Text/Title
st.title("COMET Telescope")

@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    return img

img = load_image("data/COMET_lockup-dark.png")
st.sidebar.image(img, width=200)

modelname = st.sidebar.selectbox("COMET model", COMET.available_models(), 5)
cuda = st.sidebar.checkbox("CUDA available")

st.sidebar.subheader("Perform bootstrap resampling:")
bootstrap = st.sidebar.number_input(
    "Number of samples for significance test (0 to disable)",
    min_value=0,
    max_value=1000,
    value=300,
    step=50,
)
prob_thresh = st.sidebar.slider(
    "P-value threshold for significance test:", 0.0, 1.0, value=0.5, step=0.1
)
st.sidebar.subheader("Perform robustness analysis:")
robustness_options = st.sidebar.multiselect('Robustness filters',
    ['named entities', 'glossary terms'],
)

glossary = False
if 'glossary terms' in robustness_options:
    glossary_file = st.file_uploader("Upload a glossary file", type=["txt"])
    glossary = read_lines(glossary_file)

st.sidebar.subheader("Segment length filter:")
st.sidebar.write(
    "In order to isolate segments according to caracter length "
    "we will create a density function according to the number "
    "of characters in the source. The next slider can be used to specify:"
)
st.sidebar.latex(r'''P(a \leq X \leq b) ''')

length_interval = st.sidebar.slider("Specify 'a' and 'b':",
    0, 100, step=5, value=(0, 100)
)

testset = PairedTestset.read_data()
if testset:
    corpus_size = len(testset)
    if 'named entities' in robustness_options:
        if not testset.check_stanza_languages():
            robustness_options.remove('named entities')
        testset = testset.ner_filter()
        st.success(
            "Named Entities Filter applied! " 
        )
    
    if length_interval != (0, 100):
        testset = testset.length_filter(length_interval[0], length_interval[1])
        st.success(
            "Length filter applied! "
        )
    if glossary:
        testset = testset.glossary_filter(glossary)
        st.success(
            "Glossary filter applied! "
        )
    st.success("Corpus reduced in {:.2f}%".format(1- len(testset)/corpus_size))

    if st.button("Run Comparison"):
        metrics = [COMET(modelname), BLEU(), chrF(), LengthRatio()]
        paired_results = []
        for metric in metrics:
            paired_results.append(metric.score_paired_testset(testset, cuda=True))

        st.header("Overall quality:")
        PairedResult.display_summary(paired_results)
        paired_results[0].display_distributions()
        paired_results[0].display_segment_comparison()
        
        # Bootstrap Resampling
        if bootstrap > 0:
            st.header("Bootstrap resampling results:")
            with st.spinner("Running bootstrap resampling..."):
                # We will save time by reusing the COMET scores that were precomputed.
                bootstrap_result =  metrics[0].paired_bootstrap(
                    testset, int(bootstrap), prob_thresh, 
                    precomputed_result=paired_results[0]
                )
                bootstrap_result.display_wins()
                for metric in metrics[1:]:
                    bootstrap_result =  metric.paired_bootstrap(
                        testset, int(bootstrap), prob_thresh, 
                    )
                    if bootstrap_result:
                        bootstrap_result.display_wins()
    