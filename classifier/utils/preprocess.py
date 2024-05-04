import os
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
import logging

class dfBuilder:
    # Initialize the dfBuilder class with path to data
    def __init__(self, path = "../../data/raw_data", stopword_path = "../../data/clinical-stopwords.txt"):
        self.path = path 
        self.stopwords = open(stopword_path).read().split()
        self.dir_list = os.listdir(path)
        logging.info(f"New dfBuilder class initialized with data in {self.path}")

    # Private method to build the dataframe
    def _build_df(self):
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

    # Private function to preprocess datframes 
    def _preprocess_df(self, df):
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

    # Function for user to call to generate train and test datasets 
    def train_test(self):
        self._build_df()
        self.df_train = self._preprocess_df(self.df_train)
        self.df_test = self._preprocess_df(self.df_test)
        logging.info('The train and test set have been generated. This step did not save the train and test set to CSV.')
        return self.df_train, self.df_test
    
    # Function for user to save full processed test + train set, test set and train set
    def save_to_csv(self):
        df_train, df_test = self.train_test()
        df = pd.concat([df_train, df_test], ignore_index=True)
        df_train.to_csv("../../data/train_cleaned.csv")
        df_test.to_csv("../../data/test_cleaned.csv")
        df.to_csv("../../data/data_cleaned.csv")
        logging.info("Train and test sets saved to CSV with names: train_cleaned, test_cleaned and data_cleaned.")


df = dfBuilder()
df.save_to_csv()


