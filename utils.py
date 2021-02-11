import streamlit as st
import sacrebleu
import numpy as np
import pandas as pd


@st.cache
def comet_predict(model, src, mt, ref, cuda):
    data = {"src": src, "mt": mt, "ref": ref}
    data = [dict(zip(data, t)) for t in zip(*data.values())]
    return model.predict(data, cuda=cuda, show_progress=True)


@st.cache
def run_BLEU(mt, ref):
    bleu = sacrebleu.corpus_bleu(mt, [ref])
    return bleu.score, bleu.sys_len / bleu.ref_len


@st.cache
def run_chrF(mt, ref):
    bleu = sacrebleu.corpus_chrf(mt, [ref])
    return bleu.score


def paired_bootstrap(
    references: list,
    x: list,
    y: list,
    x_comet_scores: list,
    y_comet_scores: list,
    num_samples=1000,
    sample_ratio=0.5,
) -> None:
    def update_wins(x_score, y_score, wins):
        if y_score > x_score:
            wins[1] += 1
        elif y_score < x_score:
            wins[0] += 1
        else:
            wins[2] += 1
        return wins

    comet_wins, bleu_wins, chrf_wins = ([0, 0, 0], [0, 0, 0], [0, 0, 0])
    n = len(references)
    ids = list(range(n))

    sample_size = int(n * sample_ratio)
    y_scores = {"COMET": [], "BLEU": [], "chrF": []}
    x_scores = {"COMET": [], "BLEU": [], "chrF": []}
    for _ in range(num_samples):
        # Subsample the gold and system outputs (with replacement)
        reduced_ids = np.random.choice(ids, size=sample_size, replace=True)
        # Calculate accuracy on the reduced sample and save stats
        reduced_ref = [references[i] for i in reduced_ids]
        reduced_x = [x[i] for i in reduced_ids]
        reduced_y = [y[i] for i in reduced_ids]

        x_comet = sum([x_comet_scores[i] for i in reduced_ids]) / len(reduced_ids)
        y_comet = sum([y_comet_scores[i] for i in reduced_ids]) / len(reduced_ids)
        x_bleu, _ = run_BLEU(reduced_x, reduced_ref)
        y_bleu, _ = run_BLEU(reduced_y, reduced_ref)
        x_chrF = run_chrF(reduced_x, reduced_ref)
        y_chrF = run_chrF(reduced_y, reduced_ref)
        comet_wins = update_wins(x_comet, y_comet, comet_wins)
        bleu_wins = update_wins(x_bleu, y_bleu, bleu_wins)
        chrf_wins = update_wins(x_chrF, y_chrF, chrf_wins)

        x_scores["COMET"].append(x_comet)
        x_scores["BLEU"].append(x_bleu)
        x_scores["chrF"].append(x_chrF)

        y_scores["COMET"].append(y_comet)
        y_scores["BLEU"].append(y_bleu)
        y_scores["chrF"].append(y_chrF)

    wins = {"COMET": comet_wins, "BLEU": bleu_wins, "chrF": chrf_wins}

    x_stats = {
        "COMET": (
            np.mean(x_scores["COMET"]),
            np.median(x_scores["COMET"]),
            x_scores["COMET"][int(num_samples * 0.025)],
            x_scores["COMET"][int(num_samples * 0.975)],
        ),
        "BLEU": (
            np.mean(x_scores["BLEU"]),
            np.median(x_scores["BLEU"]),
            x_scores["BLEU"][int(num_samples * 0.025)],
            x_scores["BLEU"][int(num_samples * 0.975)],
        ),
        "chrF": (
            np.mean(x_scores["chrF"]),
            np.median(x_scores["chrF"]),
            x_scores["chrF"][int(num_samples * 0.025)],
            x_scores["chrF"][int(num_samples * 0.975)],
        ),
    }

    y_stats = {
        "COMET": (
            np.mean(y_scores["COMET"]),
            np.median(y_scores["COMET"]),
            y_scores["COMET"][int(num_samples * 0.025)],
            y_scores["COMET"][int(num_samples * 0.975)],
        ),
        "BLEU": (
            np.mean(y_scores["BLEU"]),
            np.median(y_scores["BLEU"]),
            y_scores["BLEU"][int(num_samples * 0.025)],
            y_scores["BLEU"][int(num_samples * 0.975)],
        ),
        "chrF": (
            np.mean(y_scores["chrF"]),
            np.median(y_scores["chrF"]),
            y_scores["chrF"][int(num_samples * 0.025)],
            y_scores["chrF"][int(num_samples * 0.975)],
        ),
    }
    return wins, x_stats, y_stats
