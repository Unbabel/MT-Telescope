import abc
import numpy as np
import plotly.figure_factory as ff
import streamlit as st
import pandas as pd
import altair as alt
from testset import PairedTestset


class MetricResult(metaclass=abc.ABCMeta):
    def __init__(
        self, 
        sys_score: int, 
        seg_scores: list, 
        sources: list, 
        hypothesis: list, 
        references: list
    ) -> None:
        self.sys_score = sys_score
        self.seg_scores = seg_scores
        self.sources = sources
        self.references = references
        self.hypothesis = hypothesis

class PairedResult:
    def __init__(self, 
        x_result: list, 
        y_result: list,
        metric: str,
    ) -> None:
        self.x_result = x_result
        self.y_result = y_result
        assert self.x_result.sources == self.y_result.sources
        assert self.x_result.references == self.y_result.references
        self.sources = x_result.sources
        self.references = x_result.references
        self.system_x = x_result.hypothesis
        self.system_y = y_result.hypothesis
        self.metric = metric
    
    @staticmethod
    def display_summary(paired_results):
        summary = {
            "x" : [p_res.x_result.sys_score for p_res in paired_results],
            "y": [p_res.y_result.sys_score for p_res in paired_results]
        }

        st.dataframe(
            pd.DataFrame.from_dict(
                summary,
                orient="index",
                columns=[r.metric for r in paired_results],
        ))
    
    def display_distributions(self) -> None:
        scores = np.array([self.x_result.seg_scores, self.y_result.seg_scores]).T
        hist_data = [scores[:, i] for i in range(scores.shape[1])]
        fig = ff.create_distplot(
            hist_data,
            ["System X", "System Y"],
            bin_size=[0.1 for _ in range(scores.shape[1])],
        )
        st.plotly_chart(fig)

    def display_segment_comparison(self,) -> None:
        scores = np.array([self.x_result.seg_scores, self.y_result.seg_scores]).T
        chart_data = pd.DataFrame(scores, columns=["x_score", "y_score"])
        chart_data["difference"] = np.absolute(scores[:, 0] - scores[:, 1])
        chart_data["source"] = self.sources
        chart_data["reference"] = self.references
        chart_data["x"] = self.system_x
        chart_data["y"] = self.system_y

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
        
    def display_json(self) -> None:
        if self.x_result.seg_scores is None:
            data = {
                "src": self.sources, 
                "ref": self.references,
                "system_x": [{"translation": mt} for mt in self.system_x],
                "system_y": [{"translation": mt} for mt in self.system_y],
            }
            data = [dict(zip(data, t)) for t in zip(*data.values())]
            st.json(data)
        else:
            data = {
                "src": self.sources, 
                "ref": self.references,
                "system_x": [
                    {"translation": mt, "score": score} 
                    for mt, score in zip(self.system_x, self.x_result.seg_scores)
                ],
                "system_y": [
                    {"translation": mt, "score": score} 
                    for mt, score in zip(self.system_y, self.y_result.seg_scores)
                ],
            }
            data = [dict(zip(data, t)) for t in zip(*data.values())]
            st.json(data)


class BootstrapResult:
    def __init__(self, x_scores: list, y_scores: list, win_count: list, num_samples: int, metric: str):
        self.x_scores = x_scores
        self.y_scores = y_scores
        self.win_count = win_count
        self.metric = metric
        self.x_stats = {
            "mean": np.mean(self.x_scores),
            "median": np.median(self.x_scores),
            "lower_bound": self.x_scores[int(num_samples * 0.025)],
            "upper_bound": self.x_scores[int(num_samples * 0.975)],
        }
        self.y_stats = {
            "mean": np.mean(self.y_scores),
            "median": np.median(self.y_scores),
            "lower_bound": self.y_scores[int(num_samples * 0.025)],
            "upper_bound": self.y_scores[int(num_samples * 0.975)],
        }
    

    def display_wins(self):
        data = []
        metric_x_wins = self.win_count[0] / sum(self.win_count)
        metric_y_wins = self.win_count[1] / sum(self.win_count)
        metric_ties = self.win_count[2] / sum(self.win_count)
        data.append({
            "metric": self.metric,
            "x win (%)": metric_x_wins,
            "y win (%)": metric_y_wins,
            "ties (%)": metric_ties,
        })
        df = pd.DataFrame(data)
        st.dataframe(df)


    



