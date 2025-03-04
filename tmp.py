from rdkit import RDLogger
from IPython.display import HTML
from molgx import *
import pickle
from joblib import parallel_backend
import os
import joblib

import threading
from concurrent.futures import ThreadPoolExecutor

import cudf

os.environ["OMP_NUM_THREADS"] = "6"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Supress RDKit logger message
RDLogger.DisableLog('rdApp.*')
# Suppress warnings of Python library
import warnings
warnings.filterwarnings('ignore')

moldata = joblib.load("example/jupyter_notebook/moldata_features.joblib")

target_df = moldata.get_property_vector()
numeric_df = target_df.apply(pd.to_numeric, errors='coerce').dropna()
print(numeric_df)
print(len(target_df), len(numeric_df))
mask = target_df.applymap(lambda x: pd.to_numeric(x, errors='coerce')).notnull().all(axis=1)
print(mask)
target_df.loc[mask] = target_df.loc[mask].astype("float64")

target_df["mol_id"] = target_df["mol_id"].astype("string")
print(target_df)
print(target_df.dtypes)

target_df = cudf.DataFrame.from_pandas(target_df)
print(target_df)
