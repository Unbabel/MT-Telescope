import streamlit as st
import pandas as pd
import numpy as np
from comet.models import download_model
from io import StringIO
import plotly.figure_factory as ff
from PIL import Image
import altair as alt
from utils import *

COMET_MODELS = [
    "emnlp-base-da-ranker",
    "wmt-base-da-ranker-1719",
    "wmt-base-da-estimator-1718",
    "wmt-base-da-estimator-1719",
    "wmt-large-da-estimator-1718",
    "wmt-large-da-estimator-1719",
    "wmt-large-qe-estimator-1719",
    "wmt-large-hter-estimator",
    "wmt-base-hter-estimator",
]

# Text/Title
st.title("COMET Telescope")


@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    return img


def loading_model(modelname: str):
    return download_model(modelname)


@st.cache()
def read_lines(file):
    if file is not None:
        file = StringIO(file.getvalue().decode())
        lines = [line.strip() for line in file.readlines()]
        return lines
    return None


def display_distributions(scores: np.array) -> None:
    hist_data = [scores[:, i] for i in range(scores.shape[1])]
    fig = ff.create_distplot(
        hist_data,
        ["System X", "System Y"],
        bin_size=[0.1 for _ in range(scores.shape[1])],
    )
    st.plotly_chart(fig)


def display_segment_comparison(
    scores: np.array, sources: list, references: list, x: list, y: list
) -> None:
    chart_data = pd.DataFrame(scores, columns=["x_score", "y_score"])
    chart_data["difference"] = compute_score_diff(scores[:, 0], scores[:, 1])
    chart_data["source"] = sources
    chart_data["reference"] = references
    chart_data["x"] = x
    chart_data["y"] = y

    c = (
        alt.Chart(chart_data)
        .mark_circle()
        .encode(
            x="x_score",
            y="y_score",
            size="difference",
            color=alt.Color("difference"),
            tooltip=[
                "x",
                "y",
                "reference",
                "difference",
                "source",
                "x_score",
                "y_score",
            ],
        )
    )
    st.altair_chart(c, use_container_width=True)


def display_json(
    sources: list, references: list, system_outputs: list, scores: np.array
) -> None:
    data = {"src": sources, "ref": references}
    for i in range(len(mt_file)):
        data[f"hyp{i}"] = system_outputs[i]
        data[f"hyp{i}_score"] = scores[:, i].tolist()
    data = [dict(zip(data, t)) for t in zip(*data.values())]
    st.json(data)


def compute_score_diff(x: np.array, y: np.array):
    return np.absolute(x - y)


def display_stats(system_stats):
    df = pd.DataFrame(
        {
            "metric": list(system_stats.keys()),
            "mean": [system_stats[m][0] for m in system_stats.keys()],
            "median": [system_stats[m][1] for m in system_stats.keys()],
            "lower_bound": [system_stats[m][2] for m in system_stats.keys()],
            "upper_bound": [system_stats[m][3] for m in system_stats.keys()],
        }
    )
    df.set_index(["metric"])
    st.dataframe(df)


def display_wins(wins):
    data = []
    for metric in list(wins.keys()):
        metric_x_wins = wins[metric][0] / sum(wins[metric])
        metric_y_wins = wins[metric][1] / sum(wins[metric])
        metric_ties = wins[metric][2] / sum(wins[metric])
        data.append(
            {
                "metric": metric,
                "x win (%)": metric_x_wins,
                "y win (%)": metric_y_wins,
                "ties (%)": metric_ties,
            }
        )
    df = pd.DataFrame(data)
    st.dataframe(df)


img = load_image("data/COMET_lockup-dark.png")
st.sidebar.image(img, width=200)

modelname = st.sidebar.selectbox("COMET model", COMET_MODELS, 5)
cuda = st.sidebar.checkbox("CUDA available")

st.sidebar.subheader("Perform bootstrap resampling:")
bootstrap = st.sidebar.number_input(
    "Number of samples for significance test (0 to disable)",
    min_value=0,
    max_value=1000,
    step=50,
)
prob_thresh = st.sidebar.slider(
    "P-value threshold for significance test:", 0.0, 1.0, step=0.1
)

st.sidebar.subheader("Upload Files:")
source_file = st.sidebar.file_uploader("Upload Sources", type=["txt"])
sources = read_lines(source_file)

ref_file = st.sidebar.file_uploader("Upload References", type=["txt"])
references = read_lines(ref_file)

x_file = st.sidebar.file_uploader("Upload System X Translations", type=["txt"])
x = read_lines(x_file)

y_file = st.sidebar.file_uploader("Upload System Y Translations", type=["txt"])
y = read_lines(y_file)

if (
    (ref_file is not None)
    and (source_file is not None)
    and (y_file is not None)
    and (x_file is not None)
):
    system_outputs = [x, y]
    st.success("Source, Refereces and Translations were successfully uploaded!")
    if st.button("Run Comparison"):
        with st.spinner("Loading COMET model..."):
            model = loading_model(modelname)
        summary = {}
        with st.spinner("Running metrics for system X"):
            x_bleu, x_lenght_ratio = run_BLEU(x, references)
            x_chrf = run_chrF(x, references)
            _, x_scores = comet_predict(model, sources, x, references, cuda)
            x_comet = sum(x_scores) / len(x_scores)

        summary["x"] = [x_comet, x_bleu, x_chrf, x_lenght_ratio]
        with st.spinner("Running metrics for system Y"):
            y_bleu, y_lenght_ratio = run_BLEU(y, references)
            y_chrf = run_chrF(y, references)
            _, y_scores = comet_predict(model, sources, y, references, cuda)
            y_comet = sum(y_scores) / len(y_scores)

        summary["y"] = [y_comet, y_bleu, y_chrf, y_lenght_ratio]

        st.dataframe(
            pd.DataFrame.from_dict(
                summary,
                orient="index",
                columns=["COMET", "BLEU", "chrF", "Length Ratio"],
            )
        )

        display_distributions(np.array([x_scores, y_scores]).T)
        display_segment_comparison(
            scores=np.array([x_scores, y_scores]).T,
            sources=sources,
            references=references,
            x=x,
            y=y,
        )

        if bootstrap > 0:
            with st.spinner("Running bootstrap resampling..."):
                wins, x_stats, y_stats = paired_bootstrap(
                    references, x, y, x_scores, y_scores, int(bootstrap), prob_thresh
                )
                st.header("Bootstrap Resampling Results:")
                st.subheader("System X stats:")
                display_stats(x_stats)
                st.subheader("System Y stats:")
                display_stats(y_stats)
                st.subheader("Comparison stats:")
                display_wins(wins)

        data = {
            "src": sources,
            "ref": references,
            "x": x,
            "y": y,
        }
        data = [dict(zip(data, t)) for t in zip(*data.values())]
        for i in range(len(data)):
            data[i]["metrics"] = {
                "x": {"COMET": x_scores[i]},
                "y": {"COMET": y_scores[i]},
            }
        st.json(data)

else:
    st.info(
        "Please use the sidebar to upload Source, Reference and the Translations you wish to compare."
    )
