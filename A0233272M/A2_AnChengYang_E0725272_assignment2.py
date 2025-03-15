#!/usr/bin/env python.

"""
CS4248 ASSIGNMENT 2 Template

TODO: Modify the variables below.  Add sufficient documentation to cross
reference your code with your writeup.

"""
# pip install numpy pandas spacy lightgbm scikit-learn imbalanced-learn && python -m spacy download en_core_web_md

# Standard library imports
import string
from collections import Counter

# Third-party imports
import numpy as np
import pandas as pd
import spacy
import lightgbm as lgb

# Scikit-learn imports
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


# Imbalanced-learn imports
from imblearn.over_sampling import SMOTE


_NAME = "AnChengYang"
_STUDENT_NUM = 'E0725272'


nlp = spacy.load("en_core_web_md")

class EnhancedNLPFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.entity_types = None  # Store entity types seen in training
        self.dependency_types = None  # Store dependency types
        
    def fit(self, X, y=None):
        """Collects all entity types and dependency labels present in the training data."""
        entity_types_set = set()
        dependency_types_set = set()
        
        for text in X:
            doc = nlp(text)
            # Collect entity types
            for ent in doc.ents:
                entity_types_set.add(ent.label_)
            
            # Collect dependency types
            for token in doc:
                dependency_types_set.add(token.dep_)
                
        self.entity_types = sorted(entity_types_set)  # Sort to ensure consistent order
        self.dependency_types = sorted(dependency_types_set)
        return self
    
    def transform(self, X, y=None):
        """Extracts all NLP features."""
        syntactic_features = self._extract_syntactic_features(X)
        entity_features = self._extract_entity_features(X)
        dependency_features = self._extract_dependency_features(X)
        semantic_features = self._extract_semantic_features(X)
        
        # Combine all feature sets
        all_features = np.hstack((
            syntactic_features, 
            entity_features, 
            dependency_features,
            semantic_features,
        ))
        
        return all_features
    
    def _extract_syntactic_features(self, X):
        """Extract syntactic features from texts."""
        features = []
        for text in X:
            doc = nlp(text)
            sentence_length = len(doc)
            safe_divisor = max(1, sentence_length)  # Avoid division by zero
            
            # Basic syntactic features
            punctuation_count = sum(1 for token in doc if token.text in string.punctuation) / safe_divisor
            
            # POS features with normalization
            pos_counts = Counter([token.pos_ for token in doc])
            noun_count = pos_counts.get("NOUN", 0) / safe_divisor
            verb_count = pos_counts.get("VERB", 0) / safe_divisor
            adj_count = pos_counts.get("ADJ", 0) / safe_divisor
            adv_count = pos_counts.get("ADV", 0) / safe_divisor
            pron_count = pos_counts.get("PRON", 0) / safe_divisor
            det_count = pos_counts.get("DET", 0) / safe_divisor
            
            # Stopwords and punctuation
            stopword_count = sum(token.is_stop for token in doc) / safe_divisor
            
            # Sentence structure features
            avg_token_length = sum(len(token.text) for token in doc) / safe_divisor
            uppercase_ratio = sum(1 for token in doc if token.text.isupper()) / safe_divisor
            
            # Sentence count
            sentence_count = len(list(doc.sents))
            avg_words_per_sentence = sentence_count > 0 and sentence_length / sentence_count or 0
            
            features.append([
                sentence_length,
                sentence_count,
                avg_words_per_sentence,
                punctuation_count,
                noun_count,
                verb_count,
                adj_count,
                adv_count,
                pron_count,
                det_count,
                stopword_count,
                avg_token_length,
                uppercase_ratio
            ])
        return np.array(features)
    
    def _extract_entity_features(self, X):
        """Extract named entity features from texts."""
        features = []
        entity_types_list = self.entity_types or []  # Use stored entity types
        
        if not entity_types_list:  # Handle case where no entities were found during fit
            return np.zeros((len(X), 1))

        for text in X:
            doc = nlp(text)
            total_entities = len(doc.ents)
            safe_divisor = max(1, total_entities)  # Avoid division by zero
            
            # Entity type distributions
            type_counts = {etype: 0 for etype in entity_types_list}
            for ent in doc.ents:
                if ent.label_ in type_counts:
                    type_counts[ent.label_] += 1
            
            # Normalized entity counts
            normalized_counts = [type_counts[etype] / safe_divisor for etype in entity_types_list]
            
            # Entity density (entities per token)
            entity_density = total_entities / max(1, len(doc))
            
            # Entity length features
            entity_lengths = [len(ent) for ent in doc.ents]
            avg_entity_length = sum(entity_lengths) / safe_divisor if entity_lengths else 0
            
            feature_row = [total_entities, entity_density, avg_entity_length] + normalized_counts
            features.append(feature_row)

        return np.array(features)
    
    def _extract_dependency_features(self, X):
        """Extract dependency parsing features from texts."""
        features = []
        dependency_types_list = self.dependency_types or []
        
        if not dependency_types_list:
            return np.zeros((len(X), 1))
            
        for text in X:
            doc = nlp(text)
            token_count = len(doc)
            safe_divisor = max(1, token_count)
            
            # Dependency type distributions
            dep_counts = {dep: 0 for dep in dependency_types_list}
            for token in doc:
                if token.dep_ in dep_counts:
                    dep_counts[token.dep_] += 1
            
            # Normalized dependency counts
            normalized_deps = [dep_counts[dep] / safe_divisor for dep in dependency_types_list]
            
            # Tree complexity features
            root_count = sum(1 for token in doc if token.dep_ == "ROOT")
            avg_children = sum(len(list(token.children)) for token in doc) / safe_divisor
            max_depth = self._find_max_tree_depth(doc)
            
            feature_row = [root_count, avg_children, max_depth] + normalized_deps
            features.append(feature_row)
            
        return np.array(features)
    
    def _find_max_tree_depth(self, doc):
        """Calculate the maximum depth of the dependency tree."""
        if not doc:
            return 0
            
        def get_depth(token):
            if not list(token.children):
                return 0
            return 1 + max(get_depth(child) for child in token.children)
            
        roots = [token for token in doc if token.dep_ == "ROOT"]
        if not roots:
            return 0
            
        return max(get_depth(root) for root in roots)
    
    def _extract_semantic_features(self, X):
        """Extract semantic features from the text."""
        features = []
        
        for text in X:
            doc = nlp(text)
            token_count = len(doc)
            safe_divisor = max(1, token_count)
            
            # Vector statistics (if available)
            has_vectors = sum(token.has_vector for token in doc)
            vector_ratio = has_vectors / safe_divisor
            
            # Semantic similarity features
            # Average similarity between consecutive tokens
            consecutive_similarities = []
            for i in range(len(doc) - 1):
                if doc[i].has_vector and doc[i+1].has_vector:
                    consecutive_similarities.append(doc[i].similarity(doc[i+1]))
                    
            avg_similarity = sum(consecutive_similarities) / max(1, len(consecutive_similarities)) if consecutive_similarities else 0
            
            # Extract lexical cohesion based on lemmas
            unique_lemmas = set(token.lemma_ for token in doc if not token.is_stop and not token.is_punct)
            lemma_ratio = len(unique_lemmas) / safe_divisor if unique_lemmas else 0
            
            features.append([
                vector_ratio,
                avg_similarity,
                lemma_ratio
            ])
            
        return np.array(features)

# Combine TF-IDF features with syntactic and named entity features
combined_features = FeatureUnion([
    ('tfidf', TfidfVectorizer(max_features=25000, ngram_range=(1, 3))),
    ('syntactic', EnhancedNLPFeatureExtractor()),
])

def train_model(model, X_train, y_train):
    ''' TODO: train your model based on the training data '''
    print("extracting features...")
    # create features
    X_train_combined_features = combined_features.fit_transform(X_train)

    # oversample the minority class
    smote = SMOTE(random_state=42)
    X_train_combined_features_oversampled, y_train_oversampled = smote.fit_resample(X_train_combined_features, y_train)
    y_train_mapped = y_train_oversampled.values + 1
    
    print("training model...")
    # fit model
    model.fit(X_train_combined_features_oversampled, y_train_mapped)


def predict(model, X_test):
    print("predicting...")
    predictions = model.predict(X_test)
    return predictions - 1

def generate_result(test, y_pred, filename):
    ''' generate csv file base on the y_pred '''
    test['Verdict'] = pd.Series(y_pred)
    test.drop(columns=['Text'], inplace=True)
    test.to_csv(filename, index=False)

def main():
    ''' load train, val, and test data '''
    train = pd.read_csv('data/train.csv')
    X = train['Text']
    y = train['Verdict']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    print("Train shape:", X_train.shape, "Test shape:", X_val.shape)

    # model parameters from hyperparameter tuning
    model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=3, 
        random_state=42, 
        class_weight='balanced',
        colsample_bytree=0.9887783209121515,
        learning_rate=0.10442515971251674, 
        max_depth=30,
        min_child_samples=4,
        min_split_gain=0.1,
        n_estimators=789,
        num_leaves=143,
        reg_alpha=0.12850035323391018, 
        reg_lambda=0.811204176736003,
        subsample=0.9641278951487912    
    )

    train_model(model, X_train, y_train)

    # create a pipeline
    pipeline = Pipeline([
        ('features', combined_features),
        ('lgb_direct', model)
    ])
    # test your model
    y_pred = predict(pipeline, X_val)

    # Use f1-macro as the metric
    score = f1_score(y_val, y_pred, average='macro')
    print('score on validation = {}'.format(score))

    # generate prediction on test data
    test = pd.read_csv('data/test.csv')
    X_test = test['Text']
    y_pred = predict(pipeline, X_test)
    
    output_filename = f"A2_{_NAME}_{_STUDENT_NUM}.csv"
    generate_result(test, y_pred, output_filename)
    
# Allow the main class to be invoked if run as a file.
if __name__ == "__main__":
    main()
