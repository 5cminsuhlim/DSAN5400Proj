import os
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation

def build_dataframe():
    df = pd.DataFrame(columns=["label", "description", "text"])
    med_stopwords = open("../data/clinical-stopwords.txt").read().split()

    def make_df_from_dir(df):
        # Get the path of data folders
        path = "../data/raw_data"
        dir_list = os.listdir(path)
        for dirs in dir_list:
            df_text = pd.read_csv(path + '/' + dirs)
            df = pd.concat([df, df_text], ignore_index=True)
        df = df.drop(columns = ['description'])
        return df
    
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

    df = make_df_from_dir(df)
    df = preprocess_df(df)
    return df


df = build_dataframe()
df.to_csv("../data/data_final.csv")
