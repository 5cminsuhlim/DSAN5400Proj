import os
import logging
from pathlib import Path

# REUSING CODE FROM HW4
def setup_logging(model_type):
    """
    Sets up logging for main.py and stores under the `logs` directory with model_type specified in the log file name

    Args:
        model_type (str): A string label identifying the type of model (Word2Vec or Doc2Vec)
    """
    # setup logs path
    proj_dir = Path(__file__).resolve().parents[2] # top directory
    logs_dir = proj_dir / 'logs'

    # create logs directory if it doesn't exist
    if not logs_dir.exists():
        os.makedirs(logs_dir)

    # setup logs file path
    log_file_path = logs_dir / f'logs_{model_type}.txt'

    # setting up logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=log_file_path,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
def setup_data_path():
    """ 
    Sets up and return data path
    """
    script_dir = Path(__file__).resolve().parent
    data_path = script_dir.parent.parent / 'data' / 'data_cleaned.csv'
    logging.info(f"Data Path: {data_path}")
    return data_path