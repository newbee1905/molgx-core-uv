from rdkit import RDLogger
from molgx import *
import pickle

import threading
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig()

# Supress RDKit logger message
RDLogger.DisableLog('rdApp.*')
# Suppress warnings of Python library
import warnings
warnings.filterwarnings('ignore')

moldata = MolData.read_csv("qm9.csv")
moldata.print_properties()

# with open("qm9_features.pickle", "rb") as f:
# 	features = pickle.load(f)
# features_fp = moldata.merge_features(map(lambda f: f.id, features))

fs_funcs = [HeavyAtomExtractor, RingExtractor, AromaticRingExtractor, FingerPrintStructureExtractor, FeatureSumOperator]
features = []

def extract_feature(fs_func):
	global moldata

	fs = moldata.extract_features(fs_func(moldata))
	
with ThreadPoolExecutor(max_workers=8) as executor:
	features = executor.map(extract_feature, fs_funcs)

features_fp = moldata.merge_features(map(lambda f: f.id, features))
features_fp.print_features()

moldata.get_dataframe(features=features_fp)
input_size = len(moldata.get_dataframe(features=features_fp).columns)
print(input_size)

models = [
	# {
	# 	"name": "rf_lumo_model.pickle",
	# 	"model": RandomForestRegressionModel,
	# 	"args": (moldata, "lumo", features_fp),
	# },
	# {
	# 	"name": "rf_gap_model.pickle",
	# 	"model": RandomForestRegressionModel,
	# 	"args": (moldata, "gap", features_fp),
	# },
	{
		"name": "rf_homo_model.pickle",
		"model": RandomForestRegressionModel,
		"args": (moldata, "homo", features_fp),
	},

	{
		"name": "lasso_lumo_model.pickle",
		"model": LassoRegressionModel,
		"args": (moldata, "lumo", features_fp),
	},
	{
		"name": "lasso_gap_model.pickle",
		"model": LassoRegressionModel,
		"args": (moldata, "gap", features_fp),
	},
	{
		"name": "lasso_homo_model.pickle",
		"model": LassoRegressionModel,
		"args": (moldata, "homo", features_fp),
	},

	{
		"name": "en_lumo_model.pickle",
		"model": ElasticNetRegressionModel,
		"args": (moldata, "lumo", features_fp),
	},
	{
		"name": "en_gap_model.pickle",
		"model": ElasticNetRegressionModel,
		"args": (moldata, "gap", features_fp),
	},
	{
		"name": "en_homo_model.pickle",
		"model": ElasticNetRegressionModel,
		"args": (moldata, "homo", features_fp),
	},
]

def train_model(model_config):
	global moldata
	global features_fp

	print(f"Training {model['name']}")
	model = model_config["model"](*model_config["args"])
	with open(mode["name"], "wb") as f:
		pickle.dump(model, f)
	print(f"Trained {model['name']}")

with ThreadPoolExecutor(max_workers=8) as executor:
	models = executor.map(train_model, models)
