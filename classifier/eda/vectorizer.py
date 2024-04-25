### IMPORTS ###
import umap
import nltk
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
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path, index_col=0)
        self.df['tokenized_text'] = self.df['text'].apply(word_tokenize)
        self.unique_labels = self.df['label'].unique()
        
        self.model = None
        self.model_type = ""
        
        self.intra_class_cosine_sim = {}
        self.intra_class_jaccard_sim = {}
        self.inter_class_cosine_sim = {}
        self.inter_class_jaccard_sim = {}

    def document_vector(self, doc):
        pass
    
    def train_model(self, size, window, min_count, workers):
        pass

    def jaccard_similarity(self, vec1, vec2):
        bool_vec1 = vec1 > 0
        bool_vec2 = vec2 > 0
        intersection = np.sum(bool_vec1 & bool_vec2)
        union = np.sum(bool_vec1 | bool_vec2)
        return intersection / union if union != 0 else 0
    
    def jaccard_similarity_matrix(self, vectors):
        n = len(vectors)
        sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                sim = self.jaccard_similarity(vectors[i], vectors[j])
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim
        return sim_matrix

    def calculate_similarities(self):
        for label in self.unique_labels:
            vectors = self.df[self.df['label'] == label]['doc_vector'].tolist()
            if not vectors:
                continue

            # intra-class similarities
            cosine_sim_matrix = cosine_similarity(vectors)
            self.intra_class_cosine_sim[label] = np.nanmean(np.where(np.eye(len(vectors)) == 1, np.nan, cosine_sim_matrix))
            
            jaccard_sim_matrix = self.jaccard_similarity_matrix(vectors)
            self.intra_class_jaccard_sim[label] = np.nanmean(np.where(np.eye(len(vectors)) == 1, np.nan, jaccard_sim_matrix))

        # inter-class similarities
        for i in range(len(self.unique_labels)):
            for j in range(i + 1, len(self.unique_labels)):
                vectors_i = self.df[self.df['label'] == self.unique_labels[i]]['doc_vector'].tolist()
                vectors_j = self.df[self.df['label'] == self.unique_labels[j]]['doc_vector'].tolist()
                if vectors_i and vectors_j:
                    self.inter_class_cosine_sim[(self.unique_labels[i], self.unique_labels[j])] = np.mean(cosine_similarity(vectors_i, vectors_j))
                    self.inter_class_jaccard_sim[(self.unique_labels[i], self.unique_labels[j])] = np.mean(self.jaccard_similarity_matrix(vectors_i + vectors_j))
        
    def visualize_heatmap(self):
        # matrices for cosine and jaccard
        sorted_labels = sorted(self.unique_labels, key=lambda x: int(x))
        num_classes = len(sorted_labels)
        cosine_matrix = np.zeros((num_classes, num_classes))
        jaccard_matrix = np.zeros((num_classes, num_classes))
        
        # fill matrices w/ intra-class and inter-class
        for i, label_i in enumerate(sorted_labels):
            for j, label_j in enumerate(sorted_labels):
                if i == j: # intra-class similarity
                    cosine_matrix[i, j] = self.intra_class_cosine_sim.get(label_i, 0)
                    jaccard_matrix[i, j] = self.intra_class_jaccard_sim.get(label_i, 0)
                else: # inter-class similarity
                    inter_key = (label_i, label_j)
                    reverse_key = (label_j, label_i)
                    cosine_matrix[i, j] = self.inter_class_cosine_sim.get(inter_key, 
                                            self.inter_class_cosine_sim.get(reverse_key, 0))
                    jaccard_matrix[i, j] = self.inter_class_jaccard_sim.get(inter_key, 
                                            self.inter_class_jaccard_sim.get(reverse_key, 0))

        # print("Cosine Similarity Matrix:")
        # print(cosine_matrix)
        # print("Jaccard Similarity Matrix:")
        # print(jaccard_matrix)

        # plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 12))
        fig.suptitle(f'{self.model_type} Similarity Heatmaps', fontsize=16)

        # heatmap for cosine 
        sns.heatmap(cosine_matrix, annot=True, cmap='coolwarm', cbar=False,
                    xticklabels=sorted_labels, yticklabels=sorted_labels,
                    fmt=".2f", annot_kws={"size": 12}, ax=axes[0])
        axes[0].set_title('Cosine Similarity')
        axes[0].set_xlabel('Labels')
        axes[0].set_ylabel('Labels')

        # heatmap for jaccard
        sns.heatmap(jaccard_matrix, annot=True, cmap='coolwarm', cbar=False,
                    xticklabels=sorted_labels, yticklabels=sorted_labels,
                    fmt=".2f", annot_kws={"size": 12}, ax=axes[1])
        axes[1].set_title('Jaccard Similarity')
        axes[1].set_xlabel('Labels')
        axes[1].set_ylabel('Labels')

        plt.tight_layout()
        plt.show()

    def visualize_datamap(self):
        pn.extension()
        
        # dimensionality reduction for doc vectors
        doc_vectors = np.stack(self.df['doc_vector'].apply(np.array))
        use_map = umap.UMAP(metric="cosine", n_neighbors=15, min_dist=0.1, random_state=1859).fit_transform(doc_vectors)
        
        # ensure all labels are strings
        self.df['label_str'] = self.df['label'].astype(str)

        # setup hover text and marker sizes based on text length
        sizes = [np.sqrt(len(x)) / 1024 for x in self.df['text']]
        hover_text = [x[:100] + " ... trimmed" if len(x) > 100 else x for x in self.df['text']]

        # create color mapping
        unique_labels = self.df['label_str'].unique()
        num_unique_labels = len(unique_labels)
        palette = sns.color_palette("husl", num_unique_labels).as_hex()
        COLOR_KEY = {label: color for label, color in zip(unique_labels, palette)}
        
        # generate plot
        enriched_plot = tnt.BokehPlotPane(
            use_map,
            labels=self.df['label_str'].tolist(),
            hover_text=hover_text,
            marker_size=sizes,
            label_color_mapping=COLOR_KEY,
            show_legend=False,
            min_point_size=0.001,
            max_point_size=0.05,
            title=f'{self.model_type} Data Map',
        )
        # print(hover_text)
        pn.Row(enriched_plot).show()

        

### WORD2VEC VECTORIZER ###
class Word2VecVectorizer(Vectorizer):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.model_type = "Word2Vec"
        
    def document_vector(self, doc):
        doc = [word for word in doc if word in self.model.wv.index_to_key]
        return np.mean(self.model.wv[doc], axis=0) if len(doc) > 0 else np.zeros(self.model.vector_size)
    
    def train_model(self, size=100, window=5, min_count=2, workers=4):
        self.model = Word2Vec(sentences=self.df['tokenized_text'], vector_size=size, window=window, min_count=min_count, workers=workers)
        self.df['doc_vector'] = self.df['tokenized_text'].apply(self.document_vector)
        print(self.df['doc_vector'])
                
    def visualize_heatmap(self, model_type="Word2Vec"):
        self.visualize_heatmap(model_type)


### DOC2VEC VECTORIZER ###
class Doc2VecVectorizer(Vectorizer):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.model_type = "Doc2Vec"
        
    def train_model(self, size=100, window=5, min_count=2, workers=4):
        tagged_data = [TaggedDocument(words=_d, tags=[str(i)]) for i, _d in enumerate(self.df['tokenized_text'])]
        self.model = Doc2Vec(tagged_data, vector_size=size, window=window, min_count=min_count, workers=workers)
        self.df['doc_vector'] = [self.model.dv[str(i)] for i in range(len(self.df))]
        print(self.df['doc_vector'])
    
    def visualize_heatmap(self, model_type="Doc2Vec"):
        self.visualize_heatmap(model_type)