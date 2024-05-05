### IMPORTS ###
import umap
import nltk
import logging
import panel as pn
import numpy as np
import pandas as pd
import seaborn as sns
import thisnotthat as tnt
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity


### PARENT CLASS ###
class Vectorizer:
    """
    Parent class for vectorizing text documents using various embedding models

    Attributes:
        df: A pandas DataFrame containing the dataset
        model: The text embedding model (Word2Vec or Doc2Vec)
        model_type: A string label identifying the type of model (Word2Vec or Doc2Vec)
        intra_class_cosine_sim: A dictionary to store intra-class cosine similarities
        intra_class_jaccard_sim: A dictionary to store intra-class Jaccard similarities
        inter_class_cosine_sim: A dictionary to store inter-class cosine similarities
        inter_class_jaccard_sim: A dictionary to store inter-class Jaccard similarities
    """

    def __init__(self, data_path):
        """
        Initializes the Vectorizer class by reading in data and setting initial values for attributes

        Args:
            data_path (str): The path to the CSV file containing the dataset
        """
        self.df = pd.read_csv(data_path, index_col=0)
        self.df["tokenized_text"] = self.df["text"].apply(word_tokenize)
        self.unique_labels = self.df["label"].unique()

        self.model = None
        self.model_type = ""

        self.intra_class_cosine_sim = {}
        self.intra_class_jaccard_sim = {}
        self.inter_class_cosine_sim = {}
        self.inter_class_jaccard_sim = {}

    def document_vector(self, doc):
        """
        Abstract method to compute the document vector. Should be implemented by subclasses

        Args:
            doc (list): The document to vectorize in the form of a list of tokens

        Returns:
            array: The vectorized document
        """
        pass

    def train_model(self, size, window, min_count, workers):
        """
        Abstract method to train the embedding model. Should be implemented by subclasses

        Args:
            size (int): The number of dimensions of the embeddings
            window (int): The max distance between the current and predicted word within a sentence
            min_count (int): The min count of words to consider when training the model
            workers (int): The number of workers (i.e. threads) to use in training
        """
        pass

    def jaccard_similarity(self, vec1, vec2):
        """
        Calculates the Jaccard similarity between two boolean vectors

        Args:
            vec1 (array): First boolean vector
            vec2 (array): Second boolean vector

        Returns:
            float: Jaccard similarity score
        """
        bool_vec1 = vec1 > 0
        bool_vec2 = vec2 > 0
        intersection = np.sum(bool_vec1 & bool_vec2)
        union = np.sum(bool_vec1 | bool_vec2)
        return intersection / union if union != 0 else 0

    def jaccard_similarity_matrix(self, vectors):
        """
        Compute a Jaccard similarity matrix for a list of boolean vectors

        Args:
            vectors (list of arrays): A list of boolean vectors

        Returns:
            array: A symmetric matrix of Jaccard similarity scores
        """

        n = len(vectors)
        sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                sim = self.jaccard_similarity(vectors[i], vectors[j])
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim
        return sim_matrix

    def calculate_similarities(self):
        """
        Calculates intra-class and inter-class similarities using cosine and Jaccard metrics and updates the respective attribute dictionaries
        """
        logging.info("Calculating similarities...")
        for label in self.unique_labels:
            vectors = self.df[self.df["label"] == label]["doc_vector"].tolist()
            if not vectors:
                continue

            # intra-class similarities
            cosine_sim_matrix = cosine_similarity(vectors)
            self.intra_class_cosine_sim[label] = np.nanmean(
                np.where(np.eye(len(vectors)) == 1, np.nan, cosine_sim_matrix)
            )
            logging.info(
                f"Cosine similarity matrix for label {label}: {cosine_sim_matrix}"
            )

            jaccard_sim_matrix = self.jaccard_similarity_matrix(vectors)
            self.intra_class_jaccard_sim[label] = np.nanmean(
                np.where(np.eye(len(vectors)) == 1, np.nan, jaccard_sim_matrix)
            )
            logging.info(
                f"Jaccard similarity matrix for label {label}: {jaccard_sim_matrix}"
            )

        # inter-class similarities
        for i in range(len(self.unique_labels)):
            for j in range(i + 1, len(self.unique_labels)):
                vectors_i = self.df[self.df["label"] == self.unique_labels[i]][
                    "doc_vector"
                ].tolist()
                vectors_j = self.df[self.df["label"] == self.unique_labels[j]][
                    "doc_vector"
                ].tolist()
                if vectors_i and vectors_j:
                    self.inter_class_cosine_sim[
                        (self.unique_labels[i], self.unique_labels[j])
                    ] = np.mean(cosine_similarity(vectors_i, vectors_j))
                    self.inter_class_jaccard_sim[
                        (self.unique_labels[i], self.unique_labels[j])
                    ] = np.mean(self.jaccard_similarity_matrix(vectors_i + vectors_j))

        logging.info("Similarity calculations complete!")

    def visualize_heatmap(self):
        """
        Visualizes the cosine and Jaccard similarity matrices using heatmaps
        """
        # matrices for cosine and jaccard
        sorted_labels = sorted(self.unique_labels, key=lambda x: int(x))
        num_classes = len(sorted_labels)
        cosine_matrix = np.zeros((num_classes, num_classes))
        jaccard_matrix = np.zeros((num_classes, num_classes))

        # fill matrices w/ intra-class and inter-class
        for i, label_i in enumerate(sorted_labels):
            for j, label_j in enumerate(sorted_labels):
                if i == j:  # intra-class similarity
                    cosine_matrix[i, j] = self.intra_class_cosine_sim.get(label_i, 0)
                    jaccard_matrix[i, j] = self.intra_class_jaccard_sim.get(label_i, 0)
                else:  # inter-class similarity
                    inter_key = (label_i, label_j)
                    reverse_key = (label_j, label_i)
                    cosine_matrix[i, j] = self.inter_class_cosine_sim.get(
                        inter_key, self.inter_class_cosine_sim.get(reverse_key, 0)
                    )
                    jaccard_matrix[i, j] = self.inter_class_jaccard_sim.get(
                        inter_key, self.inter_class_jaccard_sim.get(reverse_key, 0)
                    )

        # plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 12))
        fig.suptitle(f"{self.model_type} Similarity Heatmaps", fontsize=16)

        # heatmap for cosine
        sns.heatmap(
            cosine_matrix,
            annot=True,
            cmap="coolwarm",
            cbar=False,
            xticklabels=sorted_labels,
            yticklabels=sorted_labels,
            fmt=".2f",
            annot_kws={"size": 12},
            ax=axes[0],
        )
        axes[0].set_title("Cosine Similarity")
        axes[0].set_xlabel("Labels")
        axes[0].set_ylabel("Labels")

        # heatmap for jaccard
        sns.heatmap(
            jaccard_matrix,
            annot=True,
            cmap="coolwarm",
            cbar=False,
            xticklabels=sorted_labels,
            yticklabels=sorted_labels,
            fmt=".2f",
            annot_kws={"size": 12},
            ax=axes[1],
        )
        axes[1].set_title("Jaccard Similarity")
        axes[1].set_xlabel("Labels")
        axes[1].set_ylabel("Labels")

        plt.tight_layout()
        plt.show()

    def visualize_datamap(self):
        """
        Visualizes the document embeddings on a 2D map using UMAP and thisnotthat and color codes the vector space representations of documents based on respective labels
        """
        pn.extension()

        # dimensionality reduction for doc vectors
        doc_vectors = np.stack(self.df["doc_vector"].apply(np.array))
        use_map = umap.UMAP(
            metric="cosine", n_neighbors=15, min_dist=0.1, random_state=1859
        ).fit_transform(doc_vectors)

        # ensure all labels are strings
        self.df["label_str"] = self.df["label"].astype(str)

        # setup hover text and marker sizes based on text length
        sizes = [np.sqrt(len(x)) / 1024 for x in self.df["text"]]
        hover_text = [
            x[:100] + " ... trimmed" if len(x) > 100 else x for x in self.df["text"]
        ]

        # create color mapping
        unique_labels = self.df["label_str"].unique()
        num_unique_labels = len(unique_labels)
        palette = sns.color_palette("husl", num_unique_labels).as_hex()
        COLOR_KEY = {label: color for label, color in zip(unique_labels, palette)}

        # generate plot
        enriched_plot = tnt.BokehPlotPane(
            use_map,
            labels=self.df["label_str"].tolist(),
            hover_text=hover_text,
            marker_size=sizes,
            label_color_mapping=COLOR_KEY,
            show_legend=False,
            min_point_size=0.001,
            max_point_size=0.05,
            title=f"{self.model_type} Data Map",
        )
        pn.Row(enriched_plot).show()


### WORD2VEC VECTORIZER ###
class Word2VecVectorizer(Vectorizer):
    """
    Child class for vectorizing text using the Word2Vec embedding model
    """

    def __init__(self, data_path):
        super().__init__(data_path)
        self.model_type = "Word2Vec"
        logging.info(
            f"Initializing {self.model_type} Vectorizer with data from {data_path}"
        )

    def document_vector(self, doc):
        doc = [word for word in doc if word in self.model.wv.index_to_key]
        return (
            np.mean(self.model.wv[doc], axis=0)
            if len(doc) > 0
            else np.zeros(self.model.vector_size)
        )

    def train_model(self, size=100, window=5, min_count=2, workers=4):
        logging.info(
            f"Training {self.model_type} Vectorizer: size={size}, window={window}, min_count={min_count}, workers={workers}"
        )
        self.model = Word2Vec(
            sentences=self.df["tokenized_text"],
            vector_size=size,
            window=window,
            min_count=min_count,
            workers=workers,
        )
        self.df["doc_vector"] = self.df["tokenized_text"].apply(self.document_vector)
        logging.info("Training {self.model_type} Vectorizer complete!")
        logging.info(
            f"{self.model_type} Vectorizer document vectors after training: {self.df['doc_vector'].head()}"
        )  # logging first few to keep logfile size under control

    def visualize_heatmap(self, model_type="Word2Vec"):
        self.visualize_heatmap(model_type)


### DOC2VEC VECTORIZER ###
class Doc2VecVectorizer(Vectorizer):
    """
    Child class for vectorizing text using the Doc2Vec embedding model
    """

    def __init__(self, data_path):
        super().__init__(data_path)
        self.model_type = "Doc2Vec"
        logging.info(
            f"Initializing {self.model_type} Vectorizer with data from {data_path}"
        )

    def train_model(self, size=100, window=5, min_count=2, workers=4):
        logging.info(
            f"Training {self.model_type} Vectorizer: size={size}, window={window}, min_count={min_count}, workers={workers}"
        )
        tagged_data = [
            TaggedDocument(words=_d, tags=[str(i)])
            for i, _d in enumerate(self.df["tokenized_text"])
        ]
        self.model = Doc2Vec(
            tagged_data,
            vector_size=size,
            window=window,
            min_count=min_count,
            workers=workers,
        )
        self.df["doc_vector"] = [self.model.dv[str(i)] for i in range(len(self.df))]
        logging.info("Training {self.model_type} Vectorizer complete!")
        logging.info(
            f"{self.model_type} Vectorizer document vectors after training: {self.df['doc_vector'].head()}"
        )  # logging first few to keep logfile size under control

    def visualize_heatmap(self, model_type="Doc2Vec"):
        self.visualize_heatmap(model_type)
