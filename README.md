# Medical Text Data Classifier

## Team Members:
- [Minsuh (Eric) Lim](https://github.com/5cminsuhlim)
- [Raunak Advani](https://github.com/raunakadvani2410)
- [Abhishek Dash](https://github.com/oasisbeatle)

# Overview
TODO: PROJECT SUMMARY + WHAT IT DOES

In this project, we utilise a corpus of biomedical documents, describing 5 different classes of patient conditions accompanied by medical texts. THe project carries out several Natural Language Processing practices such as Word2Vec and Doc2Vec vectorization, in addition to training models to predict document classes.

## Dataset
TODO: ADD BOX LINK AND STEPS TO UNZIP TO APPROPRIATE DIRECTORY

## Environment Setup
Run the command `conda env create --name NAME --file ./requirements.yml` from the top directory


## HOW TO RUN
To run the Classifier, navigate to the top directory and run `python ./classifier/bin/main.py -m [word2vec, doc2vec] -v -c [rnn, xgb, nb]`

To run unit tests, navigate to the top directory (i.e. where `pytest.ini` is located) and run `pytest`
