import altair as alt
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import streamlit as st
import torch
from comet.models import download_model
from PIL import Image

from telescope import (ALL_METRICS, COMET, FILTERS, SEGMENT_METRICS,
                       SYSTEM_METRICS, LengthFilter, Testset)
from telescope.metrics.metric import Metric
from telescope.plotting import (plot_bootstraping_result,
                                plot_bucket_comparison,
                                plot_pairwise_distributions,
                                plot_segment_comparison, plot_system_results)

# HACK to avoid reloading COMET everytime something changes
if not hasattr(st, 'comet'):
  st.comet = COMET()

comet_metric = st.comet
available_metrics = {m.name: m for m in ALL_METRICS}
available_filters = {f.name: f for f in FILTERS if f.name != "length"}

# Text/Title
st.sidebar.title("MT-Telescope LOGO")

@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    return img


st.title("Welcome to MT-Telescope!")
testset = Testset.read_data()

metrics = st.sidebar.multiselect(
    "Select the system-level metric you wish to run:",
    list(available_metrics.keys()),
    default=["COMET", "sacreBLEU", "chrF", "Length-ratio"]
)

filters = st.sidebar.multiselect(
    "Select testset filters:",
    list(available_filters.keys()),
    default=["duplicates"]
)

st.sidebar.subheader("Segment length constraints:")
st.sidebar.write(
    "In order to isolate segments according to caracter length "
    "we will create a sequence length distribution that you can constraint "
    "through it's density funcion. The next slider can be used to specify:"
)
st.sidebar.latex(r"""P(a \leq X \leq b) """)

length_interval = st.sidebar.slider(
    "Specify 'a' and 'b':", 0, 100, step=5, value=(0, 100)
)

st.sidebar.subheader("Bootstrap resampling settings:")
num_samples = st.sidebar.number_input(
    "Number of random partitions (0 to disable):",
    min_value=0,
    max_value=1000,
    value=300,
    step=50,
)
sample_ratio = st.sidebar.slider(
    "Proportion (P) of the initial sample:", 0.0, 1.0, value=0.5, step=0.1
)


if testset:
    corpus_size = len(testset)
    if "named-entities" in filters:
        ner_filter = available_filters["named-entities"](testset)
        if not ner_filter.check_stanza_languages():
            filters.remove("named-entities")

        testset = ner_filter.apply_filter()
        st.success("Named Entities Filter applied! ")

    if length_interval != (0, 100):
        testset = LengthFilter(
            testset, length_interval[0], length_interval[1]
        ).apply_filter()
        st.success("Length filter applied! ")

    if "terminology" in filters:
        testset = available_filters["terminology"](testset).apply_filter()
        st.success("Glossary filter applied! ")

    if "duplicates" in filters:
        testset = available_filters["duplicates"](testset).apply_filter()
        st.success("Duplicates removed!")

    st.success("Corpus reduced in {:.2f}%".format((1 - len(testset) / corpus_size)*100))
    _, midle, _  = st.beta_columns(3)
    if midle.button("Click here for analysis"):
        system_metrics = [available_metrics[m](lang=testset.trg_lang) for m in metrics if m != "COMET"]
        if "COMET" in metrics:
            system_metrics = [comet_metric,] + system_metrics
        
        pairwise_results = []
        for metric in system_metrics:
            pairwise_results.append(metric.pairwise_comparison(testset))
        
        st.header("Overall system quality:")
        plot_system_results(pairwise_results)
        
        if "COMET" in metrics:
            comet_result = pairwise_results[0]
        else:
            comet_result = comet_metric.pairwise_comparison(testset)
        
        st.header("Segment-level comparison:")
        plot_segment_comparison(comet_result)

        st.header("Error-type analysis:")
        plot_bucket_comparison(comet_result)

        st.header("Segment-level scores histogram:")
        plot_pairwise_distributions(comet_result)
        
        # Bootstrap Resampling
        if num_samples > 0:
            st.header("Bootstrap resampling results:")
            with st.spinner("Running bootstrap resampling..."):
                for i, metric in enumerate(system_metrics):
                    bootstrap_result = metric.bootstrap_resampling(
                        testset,
                        int(num_samples),
                        sample_ratio,
                        pairwise_results[i]
                    )
                    plot_bootstraping_result(bootstrap_result)



