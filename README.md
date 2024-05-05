# Medical Abstract Classifier

## Team Members:
- [Minsuh (Eric) Lim](https://github.com/5cminsuhlim)
- [Raunak Advani](https://github.com/raunakadvani2410)
- [Abhishek Dash](https://github.com/oasisbeatle)

# Overview
In this project, we utilize a corpus of biomedical documents, specifically the abstracts from published papers. These documents describe five different classes of patient conditions such as neoplasms, digestive, nervous, cardiovascular, and pathological. The project carries out several Natural Language Processing practices such as vectorization (Word2Vec and Doc2Vec) and classification (via Naive Bayes, XGBoost, and Recurrent Neural Networks).

# Setup

## Data
Download the shared [data](https://drive.google.com/drive/folders/14Nor5eXhxmhmxGbm_vvVddKx-X4EWnUi?usp=drive_link) folder that contains all the data used for this project. 

Once the data zip file has been downloaded, unzip the folder and drag the `data` folder to the top directory. In the command line, run `mv data/test_data tests`. 

## Environment
Run the command `conda env create --name NAME --file ./requirements.yml` from the top directory

# How to Run
To run the Classifier, navigate to the top directory and run `python ./classifier/bin/main.py -m [word2vec, doc2vec] -v -c [rnn, xgb, nb]`

To run unit tests, navigate to the top directory (i.e. where `pytest.ini` is located) and run `pytest`
