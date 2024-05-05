# ref: https://stackoverflow.com/questions/72268814/importing-python-function-from-outside-of-the-current-folder
import os
import sys
import logging
import argparse
from pathlib import Path

from utils import setup_logging, setup_data_path
from eda import Word2VecVectorizer, Doc2VecVectorizer


def main(model_type, visualize_flag, clf):
    """
    Main function to process data using Word2Vec or Doc2Vec vectorization.

    Args:
        model_type (str): Type of model to use ('word2vec' or 'doc2vec').
        visualize_flag (bool): Whether to run visualization functions.
    """
    setup_logging(model_type)
    data_path = setup_data_path()

    if model_type == "word2vec":
        vectorizer = Word2VecVectorizer(data_path)
    elif model_type == "doc2vec":
        vectorizer = Doc2VecVectorizer(data_path)
    else:
        raise ValueError("Model type must be 'word2vec' or 'doc2vec'")

    vectorizer.train_model()

    if visualize_flag:
        vectorizer.calculate_similarities()
        vectorizer.visualize_heatmap()
        vectorizer.visualize_datamap()

    # TO DO: ADD CLASSIFICATION STUFF FROM HERE + UPDATE `main` DOCSTRING


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical Text Data Classification")
    parser.add_argument(
        "-m",
        "--model",
        choices=["word2vec", "doc2vec"],
        required=True,
        help="Model type to use for processing",
    )
    parser.add_argument(
        "-v",
        "--visualize",
        action="store_true",
        help="Flag to run visualization functions",
    )
    parser.add_argument(
        "-c",
        "--classifier",
        choices=["rnn", "xgb", "nb"],
        help="Model type to use for classification",
    )
    args = parser.parse_args()

    # call main function
    main(args.model, args.visualize, args.classifier)
