# CS4248 Assignment 2: Text Classification with Advanced NLP Features

This project implements a text classification system using various Natural Language Processing (NLP) techniques and machine learning models. The system combines traditional TF-IDF features with advanced linguistic features extracted using spaCy to improve classification performance.

## Project Structure

```
.
├── data/                      # Data directory
├── models/                    # Saved model files
├── notebooks/                 # Jupyter notebooks for experimentation
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_feature_eng.ipynb
│   └── various model experiments
├── A2_AnChengYang_E0725272_assignment2.py  # Main implementation
└── README.md
```

## Features

The system implements several types of NLP features:

1. **TF-IDF Features**
   - Word and character n-grams (1-3)
   - Maximum 25,000 features

2. **Syntactic Features**
   - Part-of-speech distributions
   - Sentence structure metrics
   - Punctuation and stopword statistics

3. **Named Entity Features**
   - Entity type distributions
   - Entity density
   - Average entity length

4. **Dependency Features**
   - Dependency type distributions
   - Tree complexity metrics
   - Parse tree depth analysis

5. **Semantic Features**
   - Vector-based similarity measures
   - Lexical cohesion metrics
   - Word vector statistics

## Implementation Details

The main implementation is in `A2_AnChengYang_E0725272_assignment2.py` and consists of:

- `EnhancedNLPFeatureExtractor`: A custom scikit-learn transformer that extracts linguistic features
- Feature union pipeline combining TF-IDF and linguistic features
- Model training and prediction functions
- Utility functions for result generation

## Requirements

```
numpy
pandas
spacy
lightgbm
scikit-learn
imbalanced-learn
```

Additionally, you need to download the spaCy model:
```bash
python -m spacy download en_core_web_md
```

## Experimentation

The `notebooks/` directory contains various experiments with different models and feature combinations:

- Logistic Regression
- Support Vector Machines (SVM)
- Neural Networks
- LightGBM
- Naive Bayes
- Word2Vec embeddings

## Usage

1. Install dependencies:
```bash
pip install numpy pandas spacy lightgbm scikit-learn imbalanced-learn
python -m spacy download en_core_web_md
```

2. Run the main script:
```bash
python A2_AnChengYang_E0725272_assignment2.py
```

## Results

The system generates:
- Predictions in CSV format
- Confusion matrix visualization
- Training history plots
- Classification reports with precision, recall, and F1-score metrics 