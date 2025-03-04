from rdkit import RDLogger
from IPython.display import HTML
from molgx import *
import pickle
from joblib import parallel_backend
import os
import joblib

import threading
from concurrent.futures import ThreadPoolExecutor

os.environ["OMP_NUM_THREADS"] = "4"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Supress RDKit logger message
RDLogger.DisableLog('rdApp.*')
# Suppress warnings of Python library
import warnings
warnings.filterwarnings('ignore')

moldata = joblib.load("moldata_features.joblib")

features_fp = moldata.merged_features_list[0]

features_fp.print_features()
input_size = len(moldata.get_dataframe(features=features_fp).columns)

with parallel_backend('threading', n_jobs=4):
	# logging.info("Training Lasso Lumo")
	# lasso_lumo_model = moldata.optimize_and_select_features(LassoRegressionModel(moldata, 'lumo', features_fp))
	# joblib.dump(lasso_lumo_model, 'lasso_lumo_model.joblib')
	# logging.info("FInished training Lasso Lumo")

	# logging.info("Training Lasso Gap")
	# lasso_gap_model = moldata.optimize_and_select_features(LassoRegressionModel(moldata, 'gap', features_fp))
	# joblib.dump(lasso_gap_model, 'lasso_gap_model.joblib')
	# logging.info("Finished training Lasso Gap")
	#
	# logging.info("Training Lasso Homo")
	# lasso_homo_model = moldata.optimize_and_select_features(LassoRegressionModel(moldata, 'homo', features_fp))
	# joblib.dump(lasso_homo_model, 'lasso_homo_model.joblib')
	# logging.info("Finished training Lasso Homo")

	logging.info("Training ElasticNet Lumo")
	en_lumo_model = moldata.optimize_and_select_features(CuMLElasticNetRegressionModel(moldata, 'lumo', features_fp))
	joblib.dump(en_lumo_model, 'en_lumo_model.joblib')
	logging.info("FInished training ElasticNet Lumo")

	logging.info("Training ElasticNet Gap")
	en_gap_model = moldata.optimize_and_select_features(CuMLElasticNetRegressionModel(moldata, 'gap', features_fp))
	joblib.dump(en_gap_model, 'en_gap_model.joblib')
	logging.info("Finished training ElasticNet Gap")

	logging.info("Training ElasticNet Homo")
	en_homo_model = moldata.optimize_and_select_features(CuMLElasticNetRegressionModel(moldata, 'homo', features_fp))
	joblib.dump(en_homo_model, 'en_homo_model.joblib')
	logging.info("Finished training ElasticNet Homo")
