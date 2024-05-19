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
from __future__ import annotations

import random
import re
from dataclasses import dataclass

import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import preprocessor as p
import torch as t
from ekphrasis.classes.preprocessor import TextPreProcessor
from gensim.models.keyedvectors import KeyedVectors
from IPython.display import clear_output, display
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from torch import nn
from torch.optim import AdamW
from tqdm.notebook import tqdm
from transformers import AutoTokenizer
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaModel

# %%
pattern = re.compile(r"<hashtag>.*?</hashtag>")
txt_processor = TextPreProcessor(unpack_hashtags=True, segmenter="twitter", annotate={"hashtag"})


# %%
def extract_tokenized_hashtags(txt: str) -> list[str]:
    processed = txt_processor.pre_process_doc(txt)
    matches = pattern.findall(processed)
    return [m.lstrip("<hashtag> ").rstrip(" </hashtag>") for m in matches]


def sanitize(fields: list | None) -> list:
    if not fields:
        return []
    return [f.match for f in fields]


MAX_SEQ_LEN = 74


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class TweetTokenizer(metaclass=Singleton):
    def __init__(
        self,
        base_transformer: SentenceTransformer,
        e2v: KeyedVectors,
        tokenizer: AutoTokenizer,
    ) -> None:
        super().__init__()
        self.base_transformer = base_transformer
        self.e2v = e2v
        self.tokenizer = tokenizer

    def parse_tweet(self, tweet: str) -> tuple[str, list[str], list[str]]:
        hashtags = extract_tokenized_hashtags(tweet)
        clean_txt = p.clean(tweet)
        parsed_txt = p.parse(tweet)
        emojis = sanitize(parsed_txt.emojis)
        return clean_txt, emojis, hashtags

    def tokenize(self, tweet: str) -> tuple[t.Tensor, ...]:
        clean_txt, emojis, hashtags = self.parse_tweet(tweet)
        emoji_embeddings = np.array([self.e2v[emoji] for emoji in emojis if emoji in self.e2v])
        if np.isnan(emoji_embeddings).all():
            emoji_mean = t.tensor(np.zeros(300), dtype=t.long)
        else:
            emoji_mean = t.tensor(emoji_embeddings.mean(axis=0), dtype=t.long)
        emoji_mean = emoji_mean.unsqueeze(0)

        hashtag_embeddings = np.array([self.base_transformer.encode(tag) for tag in hashtags])
        if len(hashtag_embeddings) > 0:
            hashtag_mean = t.tensor(hashtag_embeddings.mean(axis=0), dtype=t.long)
        else:
            hashtag_mean = t.tensor(np.zeros(768), dtype=t.long)
        hashtag_mean = hashtag_mean.unsqueeze(0)

        tokens = self.tokenizer.tokenize(clean_txt)
        if len(tokens) > MAX_SEQ_LEN - 2:
            tokens = tokens[: MAX_SEQ_LEN - 2]
        tokens = [self.tokenizer.cls_token, *tokens, self.tokenizer.sep_token]
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = np.ones(len(token_ids))

        padding = np.zeros(MAX_SEQ_LEN - len(token_ids))
        token_ids = np.append(token_ids, padding)
        attention_mask = np.append(attention_mask, padding)

        token_ids = t.tensor(token_ids, dtype=t.long).unsqueeze(0)
        attention_mask = t.tensor(attention_mask, dtype=t.long).unsqueeze(0)
        return token_ids, attention_mask, emoji_mean, hashtag_mean


class XLMModel(XLMRobertaModel, metaclass=Singleton): ...


class XLMSentenceTransformer(SentenceTransformer, metaclass=Singleton): ...


class XLMAutoTokenizer(AutoTokenizer, metaclass=Singleton): ...


e2v = KeyedVectors.load_word2vec_format("emoji2vec.bin", binary=True)

# %% [markdown]
# ## Feature extractor


# %%
class FeatureExtractor(metaclass=Singleton):
    def __init__(self, tweet_tokenizer: TweetTokenizer, xlm_model: XLMModel) -> None:
        self.tweet_tokenizer = tweet_tokenizer
        self.xlm_model = xlm_model
        self.features: dict[str, t.Tensor] = {}

    def extract_features(self, X: np.ndarray) -> dict[str, t.Tensor]:
        to_extract = [x for x in X if x not in self.features]
        if len(to_extract) > 0:
            all_features = np.array(
                [
                    self._get_features(*self.tweet_tokenizer.tokenize(x)).detach().numpy()
                    for x in tqdm(to_extract, desc="Tokenization")
                ]
            )
            for i in range(len(to_extract)):
                self.features[to_extract[i]] = all_features[i].reshape((1836,))
        return self.features

    def _get_features(
        self,
        txt_token_ids: t.Tensor,
        attention_mask: t.Tensor,
        emoji_embeddings: t.Tensor,
        hashtag_embeddings: t.Tensor,
    ) -> t.Tensor:
        out = self.xlm_model(txt_token_ids, attention_mask=attention_mask)[0][:, 0, :]
        out_features = t.cat([out, emoji_embeddings, hashtag_embeddings], axis=1)
        return out_features


# %% [markdown]
# ## MLP classifier head


# %%
class TweetClassifierMLP(ClassifierMixin, BaseEstimator):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.extractor = FeatureExtractor(
            TweetTokenizer(
                XLMSentenceTransformer("xlm-r-100langs-bert-base-nli-mean-tokens"),
                e2v,
                XLMAutoTokenizer.from_pretrained(
                    "sentence-transformers/xlm-r-100langs-bert-base-nli-mean-tokens"
                ),
            ),
            XLMModel.from_pretrained(
                "sentence-transformers/xlm-r-100langs-bert-base-nli-mean-tokens"
            ),
        )
        self.classifier_head = MLPClassifier(*args, **kwargs)

    def fit(self, X: np.ndarray, y: t.Tensor) -> TweetClassifierMLP:
        self.n_classes = y.shape[1]
        features = self.extractor.extract_features(X)
        features_for_fit = np.array([features[x] for x in X])
        self.classifier_head.fit(features_for_fit, y)
        if hasattr(self.classifier_head, "classes_"):
            self.classes_ = self.classifier_head.classes_
        return self

    def predict(self, X: np.ndarray) -> t.Tensor:
        features = self.extractor.extract_features(X)
        features_for_predict = np.array([features[x] for x in X])
        preds = self.classifier_head.predict(features_for_predict)
        preds = np.array(preds).reshape((len(X), self.n_classes))
        return preds

    def predict_proba(self, X: np.ndarray) -> t.Tensor:
        features = self.extractor.extract_features(X)
        features_for_predict = np.array([features[x] for x in X])
        preds = self.classifier_head.predict_proba(features_for_predict)
        preds = np.array(preds).reshape((len(X), self.n_classes))
        return preds

    def score(self, X: np.ndarray, y: t.Tensor, *args, **kwargs) -> float:
        features = self.extractor.extract_features(X)
        features_for_score = np.array([features[x] for x in X])
        return self.classifier_head.score(features_for_score, y, *args, **kwargs)

    def get_params(self, *args, **kwargs) -> dict:
        return self.classifier_head.get_params(*args, **kwargs)


# %% [markdown]
# ## KNN classifier head


# %%
class TweetClassifierKNN(ClassifierMixin, BaseEstimator):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.extractor = FeatureExtractor(
            TweetTokenizer(
                XLMSentenceTransformer("xlm-r-100langs-bert-base-nli-mean-tokens"),
                e2v,
                XLMAutoTokenizer.from_pretrained(
                    "sentence-transformers/xlm-r-100langs-bert-base-nli-mean-tokens"
                ),
            ),
            XLMModel.from_pretrained(
                "sentence-transformers/xlm-r-100langs-bert-base-nli-mean-tokens"
            ),
        )
        self.classifier_head = KNeighborsClassifier(*args, **kwargs)

    def fit(self, X: np.ndarray, y: t.Tensor) -> TweetClassifierKNN:
        self.n_classes = y.shape[1]
        features = self.extractor.extract_features(X)
        features_for_fit = np.array([features[x] for x in X])
        self.classifier_head.fit(features_for_fit, y)
        if hasattr(self.classifier_head, "classes_"):
            self.classes_ = self.classifier_head.classes_
        return self

    def predict(self, X: np.ndarray) -> t.Tensor:
        features = self.extractor.extract_features(X)
        features_for_predict = np.array([features[x] for x in X])
        preds = self.classifier_head.predict(features_for_predict)
        preds = np.array(preds).reshape((len(X), self.n_classes))
        return preds

    def predict_proba(self, X: np.ndarray) -> t.Tensor:
        features = self.extractor.extract_features(X)
        features_for_predict = np.array([features[x] for x in X])
        preds = self.classifier_head.predict_proba(features_for_predict)
        preds = np.array(preds).reshape((len(X), self.n_classes))
        return preds

    def score(self, X: np.ndarray, y: t.Tensor, *args, **kwargs) -> float:
        features = self.extractor.extract_features(X)
        features_for_score = np.array([features[x] for x in X])
        return self.classifier_head.score(features_for_score, y, *args, **kwargs)

    def get_params(self, *args, **kwargs) -> dict:
        return self.classifier_head.get_params(*args, **kwargs)


# %% [markdown]
# ## Binary relevance with MLP


# %%
class TweetClassifierBRMLP(TweetClassifierMLP):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        if "estimator" not in kwargs:
            mlp = MLPClassifier(*args, **kwargs)
            self.classifier_head = OneVsRestClassifier(mlp)
        else:
            self.classifier_head = OneVsRestClassifier(kwargs["estimator"])


# %% [markdown]
# ## Binary relevance with KNN


# %%
class TweetClassifierBRKNN(TweetClassifierKNN):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        if "estimator" not in kwargs:
            knn = KNeighborsClassifier(*args, **kwargs)
            self.classifier_head = OneVsRestClassifier(knn)
        else:
            self.classifier_head = OneVsRestClassifier(kwargs["estimator"])


# %% [markdown]
# ## Classifier chain with MLP


# %%
class TweetClassifierCCMLP(TweetClassifierMLP):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        if "base_estimator" not in kwargs:
            mlp = MLPClassifier(*args, **kwargs)
            self.classifier_head = ClassifierChain(mlp)
        else:
            classifier = kwargs.pop("base_estimator")
            self.classifier_head = ClassifierChain(classifier, **kwargs)


# %% [markdown]
# ## Classifier chain with KNN


# %%
class TweetClassifierCCKNN(TweetClassifierKNN):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        if "base_estimator" not in kwargs:
            knn = KNeighborsClassifier(*args, **kwargs)
            self.classifier_head = ClassifierChain(knn)
        else:
            self.classifier_head = ClassifierChain(kwargs["base_estimator"])
