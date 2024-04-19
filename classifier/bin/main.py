# ref: https://stackoverflow.com/questions/72268814/importing-python-function-from-outside-of-the-current-folder
import sys
import os

### SET UP ###
print("Current Working Directory:", os.getcwd())

# add the parent directory to sys.path so Python can find the utils module
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '..'))


### MAIN SCRIPT ###
from eda import Word2VecVectorizer, Doc2VecVectorizer


# using full cleaned dataset for visualizations
data_path = os.path.join(script_dir, '..', '..', 'data', 'data_cleaned.csv')
print("Data Path:", data_path)

word2vec_processor = Word2VecVectorizer(data_path)
word2vec_processor.train_model(size=100, window=5, min_count=2, workers=4)
# word2vec_processor.calculate_similarities()
# word2vec_processor.visualize_heatmap()
word2vec_processor.visualize_datamap()

# doc2vec_processor = Doc2VecVectorizer(path)
# doc2vec_processor.train_model(size=100, window=5, min_count=2, workers=4)
# doc2vec_processor.calculate_similarities()
# doc2vec_processor.visualize_heatmap()
# doc2vec_processor.visualize_datamap()
