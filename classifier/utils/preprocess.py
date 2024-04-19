import os
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation

def build_dataframe():
    df_test = pd.DataFrame(columns=["condition_label", "medical_abstract"])
    df_train = pd.DataFrame(columns=["condition_label", "medical_abstract"])
    med_stopwords = open("../../data/clinical-stopwords.txt").read().split()
    label_map = {1: 'neoplasms',  2: 'digestive', 3: 'nervous', 4:'cardiovascular', 5:'pathological'}

    def make_df_from_dir(df_train, df_test):
        # Get the path of data folders
        path = "../data/raw_data"
        dir_list = os.listdir(path)
        for dirs in dir_list:
            df_text = pd.read_csv(path + '/' + dirs)
            if 'test' in dirs:
                df_test = pd.concat([df_test, df_text], ignore_index=True)
                df_test = df_test.rename(columns={"condition_label":"label", "medical_abstract":"text"})
                df_test['label_str'] = df_test['label'].map(label_map)
            elif 'train' in dirs:
                df_train = pd.concat([df_train, df_text], ignore_index=True)
                df_train = df_train.rename(columns={"condition_label":"label", "medical_abstract":"text"})
                df_train['label_str'] = df_train['label'].map(label_map)
        return df_train, df_test

    def preprocess_df(df):
        eng_stopwords = set(stopwords.words('english'))
        numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
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
                if(x not in med_stopwords or x not in eng_stopwords):
                    if(any(map(str.isdigit, x)) == False):
                        cleaned_text.append(x)
            df.iloc[i, 1] = ' '.join(cleaned_text)
        return df

    df_train, df_test = make_df_from_dir(df_train, df_test)
    df_train = preprocess_df(df_train)
    df_test = preprocess_df(df_test)
    return df_train, df_test


df_train, df_test = build_dataframe()
df = pd.concat([df_train, df_test], ignore_index=True)
df_train.to_csv("../../data/train_cleaned.csv")
df_test.to_csv("../../data/test_cleaned.csv")
df = df.to_csv("../../data/data_cleaned.csv")