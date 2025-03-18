# MLMLC: Multi-Label Machine Learning Classification

## Project Overview

MLMLC is a machine learning project focused on multi-label classification of tweets. It uses various classification models to categorize tweets based on their content, including text, emojis, and hashtags.

## Features

- Tweet text preprocessing and tokenization
- Feature extraction from tweets (text, emojis, hashtags)
- Multiple classification approaches:
  - Binary Relevance (BR) approach
  - Classifier Chains (CC) approach
- Model implementations:
  - MLP (Multi-Layer Perceptron) classifiers
  - KNN (K-Nearest Neighbors) classifiers
- Statistical analysis of model performance

## Project Structure

- `calculate_statistics.py`: Statistical analysis of model performance
- `load_models.py`: Utility for loading saved models
- `sklearn_models/`: Directory containing model implementations
  - `models.py`: Contains classifier implementations
  - `runner.py`: Script for training and evaluating models
- `data/`: Contains training, development, and test datasets

## Data

The project uses the 2018 E-c-En dataset which contains tweets for multi-label classification:
- `2018-E-c-En-train.txt`: Training dataset
- `2018-E-c-En-dev.txt`: Development dataset
- `2018-E-c-En-test.txt`: Test dataset

## Machine Learning Approach

The project implements several classification approaches:
1. Feature extraction using pre-trained embeddings (including emoji2vec)
2. XLM-RoBERTa for text representation
3. Multi-label classification using both MLP and KNN with:
   - Binary Relevance approach (independent classifiers for each label)
   - Classifier Chains approach (accounts for label dependencies)

## Usage

To train and evaluate models:

```python
from sklearn_models.runner import train_and_evaluate

# Train and evaluate models
results = train_and_evaluate()
```

To load pre-trained models:
```python
from load_models import load_models

paths = {
    "mlp": "./mlp-TIMESTAMP.pkl",
    "knn": "./knn-TIMESTAMP.pkl",
}
models = load_models(paths)
```

To analyze model performance:
```python
from calculate_statistics import get_means_and_stddevs, test_classifier_scores

# Calculate statistics
stats = get_means_and_stddevs(scores_df)

# Compare classifiers
comparison = test_classifier_scores(classifier_a_scores, classifier_b_scores)
```

## Requirements

- Python 3.8+
- Libraries:
  - numpy
  - pandas
  - scikit-learn
  - torch
  - transformers
  - sentence-transformers
  - gensim
  - ekphrasis
  - tqdm
  - matplotlib
  - datasets
