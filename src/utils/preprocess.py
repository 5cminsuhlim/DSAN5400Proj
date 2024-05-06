import os
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
import logging

class dfBuilder:
    """
    Class to build test-train-dataframes based on biomedical data 
        
    Attributes: 
        path: contains the path to raw tsv/csv data 
        stopword_path: folder path to link to clinical stopwords 
        dir_list: list of folder directories that includes raw test and train data 
        df_train: training dataframe
        df_test: testing dataframe
    """
    
    def __init__(self, path = "../../data/raw_data", stopword_path = "../../data/clinical-stopwords.txt"):
        """
        Initializer method for the dfBuilder class for a particular directory 

        Keyword args: 
            path: path to raw tsv/csv data. default: "../../data/raw_data"
            stopword_path: path to clinical stopwords file. default: "../../data/clinical-stopwords.txt"
        """
        self.path = path 
        self.stopwords = open(stopword_path).read().split()
        self.dir_list = os.listdir(path)
        logging.info(f"New dfBuilder class initialized with data in {self.path}")

    def _build_df(self):
        """
        Devloper method to build the test and train dataframes
        """
        df_test = pd.DataFrame(columns=["condition_label", "medical_abstract"])
        df_train = pd.DataFrame(columns=["condition_label", "medical_abstract"])
        label_map = {1: 'neoplasms',  2: 'digestive', 3: 'nervous', 4:'cardiovascular', 5:'pathological'}

        for dirs in self.dir_list:
            df_text = pd.read_csv(self.path + '/' + dirs)
            if 'test' in dirs:
                df_test = pd.concat([df_test, df_text], ignore_index=True)
                df_test = df_test.rename(columns={"condition_label":"label", "medical_abstract":"text"})
                df_test['label_str'] = df_test['label'].map(label_map)
            elif 'train' in dirs:
                df_train = pd.concat([df_train, df_text], ignore_index=True)
                df_train = df_train.rename(columns={"condition_label":"label", "medical_abstract":"text"})
                df_train['label_str'] = df_train['label'].map(label_map)
        self.df_train = df_train
        self.df_test = df_test

    def _preprocess_df(self, df):
        """
        Developer function to complete NLP preprocessing tasks on a dataframe including stopword/punctuation removal

        Args:
            df: dataframe to be preprocessed and cleaned
        """
        eng_stopwords = set(stopwords.words('english'))
        for i in range(0, len(df)):
            text = df.loc[i, "text"]
            # remove punctuation
            for punct in punctuation:
                if(punct not in "'"):
                    text = text.replace(punct, " ")
                else:
                    text = text.replace(punct, "")
            # tokenize and lowercase the text
            tokenized_text = word_tokenize(text.lower())
            # remove numbers and stopwords
            cleaned_text = []
            for x in tokenized_text:
                if(x not in self.stopwords or x not in eng_stopwords):
                    if(any(map(str.isdigit, x)) == False):
                        cleaned_text.append(x)
            df.iloc[i, 1] = ' '.join(cleaned_text)
        return df
 
    def train_test(self):
        """
        User method to call the build_df() and preprocess_df() developer functions

        Returns: 
            df_train: dataframe of train set 
            df_test: dataframe of test set 
        """
        self._build_df()
        self.df_train = self._preprocess_df(self.df_train)
        self.df_test = self._preprocess_df(self.df_test)
        logging.info('The train and test set have been generated. This step did not save the train and test set to CSV.')
        return self.df_train, self.df_test
    
    def save_to_csv(self):
        """
        User method to save the cleaned test, train and complete datasets to file.
        Default save location: "../../data/"
        """
        df_train, df_test = self.train_test()
        df = pd.concat([df_train, df_test], ignore_index=True)
        df_train.to_csv("../../data/train_cleaned.csv")
        df_test.to_csv("../../data/test_cleaned.csv")
        df.to_csv("../../data/data_cleaned.csv")
        logging.info("Train and test sets saved to CSV with names: train_cleaned, test_cleaned and data_cleaned.")


