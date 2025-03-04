
from rdkit import RDLogger
from IPython.display import HTML
from molgx import *
import pickle
from joblib import parallel_backend
import os
import joblib

import threading
from concurrent.futures import ThreadPoolExecutor

os.environ["OMP_NUM_THREADS"] = "10"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Supress RDKit logger message
RDLogger.DisableLog('rdApp.*')
# Suppress warnings of Python library
import warnings
warnings.filterwarnings('ignore')

logging.info("Moving RandomForest Lumo")
with open("rf_lumo_model.pickle", "rb") as f:
	rf_lumo_model = pickle.load(f)
joblib.dump(rf_lumo_model, 'rf_lumo_model.joblib')
logging.info("FInished training RandomForest Lumo")

logging.info("Moving RandomForest Gap")
with open("rf_gap_model.pickle", "rb") as f:
	rf_gap_model = pickle.load(f)
joblib.dump(rf_gap_model, 'rf_gap_model.joblib')
logging.info("Finished training RandomForest Gap")

logging.info("Moving RandomForest Homo")
with open("rf_homo_model.pickle", "rb") as f:
	rf_homo_model = pickle.load(f)
joblib.dump(rf_homo_model, 'rf_homo_model.joblib')
logging.info("Finished training RandomForest Homo")
