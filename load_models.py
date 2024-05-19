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
import sys

import dill

sys.path.append("../")

# %% [markdown]
# ## Check params for models

# %%
paths = {
    # "first_mlp": "./mlp-2024-05-19 00:19:22.273185.pkl",
    # "first_knn": "./knn-2024-05-19 00:19:22.273185.pkl",
    # "second_mlp": "./mlp-2024-05-19 02:17:04.026831.pkl",
    # "second_knn": "./knn-2024-05-19 02:17:04.026831.pkl",
    "third_mlp": "./mlp-2024-05-19 18:23:36.996852.pkl",
    "third_knn": "./knn-2024-05-19 18:23:36.996852.pkl",
}


def load_models(paths: dict) -> dict:
    models = {}
    for name, path in paths.items():
        with open(path, "rb") as f:
            models[name] = dill.load(f)
    return models


# %%
models = load_models(paths)

# %%
models["third_mlp"].classifier_head.hidden_layer_sizes

# %%
models["third_knn"].classifier_head.n_neighbors
