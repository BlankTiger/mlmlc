# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:hydrogen
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, shapiro, ttest_ind, ttest_rel

# %%
scores_first = pd.read_csv("./scores-2024-05-19 00:19:22.273185.csv")
scores_second = pd.read_csv("./scores-2024-05-19 02:17:04.026831.csv")
scores_third = pd.read_csv("./scores-2024-05-19 18:23:36.996852.csv")
mapping = {"Unnamed: 0": "categories"}
scores_first = scores_first.rename(columns=mapping).set_index("categories")
scores_second = scores_second.rename(columns=mapping).set_index("categories")
scores_third = scores_third.rename(columns=mapping).set_index("categories")


# %%
def str_lists_to_arrays(scores: pd.DataFrame) -> pd.DataFrame:
    def parse(list_str: str) -> list[float]:
        list_str = (
            list_str.replace("\n", "")
            .replace("   ", " ")
            .replace("  ", " ")
            .replace("\t", " ")
            .strip("[]")
            .split(" ")
        )
        nums = [float(x) for x in list_str if x]
        return nums

    output = []
    for x, y in scores.items():
        output.append(parse(y))
    return output


scores_first = scores_first.apply(str_lists_to_arrays)
scores_second = scores_second.apply(str_lists_to_arrays)
scores_third = scores_third.apply(str_lists_to_arrays)

# %%
scores_first

# %%
scores_first.describe()

# %%
scores_first.info()

# %%
mlp_scores = scores_first["mlp"]
knn_scores = scores_first["knn"]
mlp_scores


# %% [markdown]
# ## Get means and stddevs


# %%
def get_means_and_stddevs(scores_df: pd.DataFrame) -> pd.DataFrame:
    output = {}
    for model_name, scores in scores_df.items():
        for_model = {}
        for category, score in scores.items():
            for_model[category] = [np.mean(score), np.std(score)]
        output[model_name] = for_model
    df = pd.DataFrame(output).T.rename(columns={0: "mean", 1: "stddev"})
    return df


# %% [markdown]
# ## Calculate statistics


# %%
def test_classifier_scores(
    classifier_a_scores: pd.Series,
    classifier_b_scores: pd.Series,
) -> pd.DataFrame:
    output = {}
    for (category, score_a), (_, score_b) in zip(
        classifier_a_scores.items(), classifier_b_scores.items()
    ):
        shapiro_score_a = shapiro(score_a)
        shapiro_score_b = shapiro(score_b)
        ttest_ind_score = ttest_ind(score_a, score_b)
        ttest_rel_score = ttest_rel(score_a, score_b)
        anova_score = f_oneway(score_a, score_b)
        output[category] = [
            shapiro_score_a,
            shapiro_score_b,
            ttest_ind_score,
            ttest_rel_score,
            anova_score,
        ]
    df = pd.DataFrame(output).T.rename(
        columns={0: "shapiro A", 1: "shapiro B", 2: "ttest ind", 3: "ttest_rel", 4: "anova"}
    )
    return df


def get_test_scores_for(scores: pd.DataFrame) -> dict:
    model_combinations = list(combinations([scores_first[model] for model in scores_first], 2))
    model_combinations_names = [
        f"{first.name} vs {second.name}" for first, second in model_combinations
    ]
    return {
        name: test_classifier_scores(*combination)
        for name, combination in zip(model_combinations_names, model_combinations)
    }


# %% [markdown]
# ## Comparing models from first run, MLP params (200, 200), knn n_neigh = 5

# %%
get_means_and_stddevs(scores_first)

# %%
get_test_scores_for(scores_first)

# %% [markdown]
# ## Comparing models from second run, MLP params (500, 300), knn n_neigh = 10

# %%
get_means_and_stddevs(scores_second)

# %%
get_test_scores_for(scores_second)

# %% [markdown]
# ## Comparing models from third run, MLP params (800, 800, 300), knn n_neigh = 3

# %%
get_means_and_stddevs(scores_third)

# %%
get_test_scores_for(scores_third)
