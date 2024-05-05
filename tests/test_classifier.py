import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from classifier.vectorizer import Word2VecVectorizer, Doc2VecVectorizer

### SET UP ###
@pytest.fixture
def test_data():
    data_path = Path(__file__).parent / 'data/test.csv'
    return data_path
@pytest.fixture
def w2v_vectorizer(test_data):
    return Word2VecVectorizer(test_data)
@pytest.fixture
def d2v_vectorizer(test_data):
    return Doc2VecVectorizer(test_data)

### TEST VECTORIZER CLASS INITIALIZATION ###
def test_initialization(w2v_vectorizer, d2v_vectorizer):
    assert isinstance(w2v_vectorizer.df, pd.DataFrame)
    assert w2v_vectorizer.model_type == "Word2Vec"
    assert isinstance(d2v_vectorizer.df, pd.DataFrame)
    assert d2v_vectorizer.model_type == "Doc2Vec"
    
### TEST MODEL TRAINING ###
def test_model_training_w2v(w2v_vectorizer):
    w2v_vectorizer.train_model()
    assert 'doc_vector' in w2v_vectorizer.df.columns

def test_model_training_d2v(d2v_vectorizer):
    d2v_vectorizer.train_model()
    assert 'doc_vector' in d2v_vectorizer.df.columns

### TEST SIMILARITY MEASURE CALCULATIONS ###
def test_calculate_similarities_w2v(w2v_vectorizer):
    w2v_vectorizer.train_model()
    w2v_vectorizer.calculate_similarities()
    assert len(w2v_vectorizer.intra_class_cosine_sim) > 0
    assert len(w2v_vectorizer.intra_class_jaccard_sim) > 0
    assert len(w2v_vectorizer.inter_class_cosine_sim) > 0
    assert len(w2v_vectorizer.inter_class_jaccard_sim) > 0
    
def test_calculate_similarities_d2v(d2v_vectorizer):
    d2v_vectorizer.train_model()
    d2v_vectorizer.calculate_similarities()
    assert len(d2v_vectorizer.intra_class_cosine_sim) > 0
    assert len(d2v_vectorizer.intra_class_jaccard_sim) > 0
    assert len(d2v_vectorizer.inter_class_cosine_sim) > 0
    assert len(d2v_vectorizer.inter_class_jaccard_sim) > 0