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
import random
import sys
import warnings

import numpy as np
import pandas as pd
import torch as t
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import cross_validate
from sklearn.utils import shuffle

sys.path.append("../")

from sklearn_models.models import (
    TweetClassifierBRKNN,
    TweetClassifierBRMLP,
    TweetClassifierCCKNN,
    TweetClassifierCCMLP,
    TweetClassifierKNN,
    TweetClassifierMLP,
)

# %%
data = pd.read_csv("data/2018-E-c-En-train.txt", sep="\t")
data_2 = pd.read_csv("data/2018-E-c-En-dev.txt", sep="\t")
data = pd.concat([data, data_2])

# %%
data = data.drop_duplicates()
data = data.dropna()


# %%
def set_rng_seed(seed: int) -> None:
    t.backends.cudnn.deterministic = True
    t.backends.cudnn.benchmark = False
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


seed = 42
set_rng_seed(seed)

# %%
labels = [
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "love",
    "optimism",
    "pessimism",
    "sadness",
    "surprise",
    "trust",
]
label_encoding = {label: i for i, label in enumerate(labels)}
X, y = data["Tweet"].to_numpy(), t.Tensor(np.array([data[label] for label in labels]).T)
# cut = 50
# X, y = X[:cut], y[:cut]
X, y = shuffle(X, y, random_state=seed)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# %% [markdown]
# ## Instantiate models for MLP and KNN classifier heads

# %%
hidden_layer_sizes = (800, 800, 300)
n_neighbors = 3
mlp_classifier = TweetClassifierMLP(hidden_layer_sizes=hidden_layer_sizes)
knn_classifier = TweetClassifierKNN(n_neighbors=n_neighbors)
br_mlp_classifier = TweetClassifierBRMLP(hidden_layer_sizes=hidden_layer_sizes)
br_knn_classifier = TweetClassifierBRKNN(n_neighbors=n_neighbors)
cc_mlp_classifier = TweetClassifierCCMLP(hidden_layer_sizes=hidden_layer_sizes)
cc_knn_classifier = TweetClassifierCCKNN(n_neighbors=n_neighbors)

# %%
chosen_metrics = [
    "accuracy",
    "f1_macro",
    "f1_micro",
    "f1_weighted",
    "precision_macro",
    "precision_micro",
    "precision_weighted",
    "recall_macro",
    "recall_micro",
    "recall_weighted",
]
cv = 10

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# %% [markdown]
# ### Preprocess all tweets

# %%
import dill

with open("extractor.pkl", "rb") as f:
    extractor = dill.load(f)
    mlp_classifier.extractor.features = extractor.features
# extractor = mlp_classifier.extractor
features = extractor.extract_features(X)

# %%
# import dill
# with open("extractor.pkl", "wb") as f:
#     dill.dump(extractor, f)

# %% [markdown]
# ### MLP cross validation

# %%
scores_mlp = cross_validate(mlp_classifier, X, y, cv=cv, scoring=chosen_metrics)

# %%
scores_mlp

# %% [markdown]
# ### KNN cross validation

# %%
scores_knn = cross_validate(knn_classifier, X, y, cv=cv, scoring=chosen_metrics)

# %%
scores_knn

# %% [markdown]
# ### MLP BR cross validation

# %%
scores_br_mlp = cross_validate(br_mlp_classifier, X, y, cv=cv, scoring=chosen_metrics)

# %%
scores_br_mlp


# %% [markdown]
# ### KNN BR cross validation

# %%
scores_br_knn = cross_validate(br_knn_classifier, X, y, cv=cv, scoring=chosen_metrics)

# %%
scores_br_knn

# %% [markdown]
# ### MLP CC cross validation

# %%
scores_cc_mlp = cross_validate(cc_mlp_classifier, X, y, cv=cv, scoring=chosen_metrics)

# %%
scores_cc_mlp


# %% [markdown]
# ### KNN CC cross validation

# %%
scores_cc_knn = cross_validate(cc_knn_classifier, X, y, cv=cv, scoring=chosen_metrics)

# %%
scores_cc_knn


# %% [markdown]
# ### Fitting

# %%
knn_classifier.fit(X, y)

# %%
mlp_classifier.fit(X, y)

# %%
br_knn_classifier.fit(X, y)

# %%
br_mlp_classifier.fit(X, y)

# %%
cc_knn_classifier.fit(X, y)

# %%
cc_mlp_classifier.fit(X, y)


# %%
def prediction_to_labels(pred: t.Tensor) -> list[str]:
    pred = pred.reshape((11,))
    return [labels[i] for i, p in enumerate(pred) if p == 1]


def get_predictions_from_classifiers(classifiers: dict, tweet: str) -> dict:
    return {
        name: prediction_to_labels(classifier.predict([tweet]))
        for name, classifier in classifiers.items()
    }


# %%
classifiers = {
    "mlp": mlp_classifier,
    "knn": knn_classifier,
    "br_mlp": br_mlp_classifier,
    "br_knn": br_knn_classifier,
    "cc_mlp": cc_mlp_classifier,
    "cc_knn": cc_knn_classifier,
}

scores = {
    "mlp": scores_mlp,
    "knn": scores_knn,
    "br_mlp": scores_br_mlp,
    "br_knn": scores_br_knn,
    "cc_mlp": scores_cc_mlp,
    "cc_knn": scores_cc_knn,
}

# %%
get_predictions_from_classifiers(
    classifiers,
    "A social media platform’s policies are good if the most extreme 10% on left and right are equally unhappy",
)

# %%
from datetime import datetime

import dill

# %%
now = datetime.now()

for name, classifier in classifiers.items():
    with open(f"{name}-{now}.pkl", "wb") as f:
        dill.dump(classifier, f)

# %%
scores_df = pd.DataFrame(scores)
scores_df.to_csv(f"scores-{now}.csv")
