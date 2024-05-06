# ref: https://stackoverflow.com/questions/72268814/importing-python-function-from-outside-of-the-current-folder
import os
import sys
import logging
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from utils import setup_logging, setup_data_path
from eda import Word2VecVectorizer, Doc2VecVectorizer


def main(model_type, visualize_flag, clf):
    """
    Main function to process data using Word2Vec or Doc2Vec vectorization.

    Args:
        model_type (str): Type of model to use ('word2vec' or 'doc2vec').
        visualize_flag (bool): Whether to run visualization functions.
        clf (str): Type of classifier to use ('nb' or 'rnn' or 'xgb')
    """
    setup_logging(model_type)
    data_path = setup_data_path()

    if model_type == "word2vec":
        vectorizer = Word2VecVectorizer(data_path)
        # open rnn model results
        if clf == "rnn":
            try:
                with open("../../assets/model_results_rnn_word2vec.txt", "r") as f:
                    contents = f.read()
                    print(contents)
            except FileNotFoundError:
                print("Failed to open model results")

            # open training plot
            try:
                img = Image.open("../../assets/model_results_word2vec.png")
                plt.imshow(img)
                plt.axis("off")
                plt.title("Model Results - Word2Vec")
                plt.show()
            except FileNotFoundError:
                print(f"Failed to find or open the image")

        elif clf == "nb":
            try:
                # open nb results
                with open("../../assets/nb_results_word2vec.txt", "r") as f:
                    contents = f.read()
                    print(contents)
            except FileNotFoundError:
                print("Failed to open model results")
        elif clf == "xgb":
            try:
                # open xgb results
                with open("../../assets/xgb_results_word2vec.txt", "r") as f:
                    contents = f.read()
                    print(contents)
            except FileNotFoundError:
                print("Failed to open model results")

        else:
            print("Error. Enter a valid model type ['xgb', 'rnn', 'nb']")
    elif model_type == "doc2vec":
        # doc2vec vectorizer
        vectorizer = Doc2VecVectorizer(data_path)
        if clf == "rnn":
            try:
                # open rnn results
                with open("../../assets/model_results_rnn_doc2vec.txt", "r") as f:
                    contents = f.read()
                    print(contents)
            except FileNotFoundError:
                print("Failed to open model results")

            try:
                # open training plot
                img = Image.open("../../assets/model_results_doc2vec.png")
                plt.imshow(img)
                plt.axis("off")
                plt.title("Model Results - Doc2Vec")
                plt.show()
            except FileNotFoundError:
                print(f"Failed to find or open the image")

        elif clf == "nb":
            try:
                # open nb results
                with open("../../assets/nb_results_doc2vec.txt", "r") as f:
                    contents = f.read()
                    print(contents)
            except FileNotFoundError:
                print("Failed to open model results")
        elif clf == "xgb":
            try:
                # open xgb results
                with open("../../assets/xgb_results_doc2vec.txt", "r") as f:
                    contents = f.read()
                    print(contents)
            except FileNotFoundError:
                print("Failed to open model results")

        else:
            raise ValueError("Error. Enter a valid model type ['xgb', 'rnn', 'nb']")
    else:
        # error handling
        raise ValueError("Model type must be 'word2vec' or 'doc2vec'")

    vectorizer.train_model()

    if visualize_flag:
        vectorizer.calculate_similarities()
        vectorizer.visualize_heatmap()
        vectorizer.visualize_datamap()


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
