from rdkit import RDLogger
from IPython.display import HTML
from molgx import *

logging.basicConfig()

# Supress RDKit logger message
RDLogger.DisableLog('rdApp.*')
# Suppress warnings of Python library
import warnings
warnings.filterwarnings('ignore')

moldata = MolData.read_csv('sample_data/QM9_partial_500.csv')
moldata.print_properties()

# generate a mask molecules whose property 'mu' is less than 5.0, and set it to a moldata.
df = moldata.get_dataframe(property=True)
moldata.set_mols_mask(df['mu'] > 5.0)

# get a dataframe with masked molecules hidden by option 'widh_mask=True'
df = moldata.get_dataframe(mols=True, smiles=True, property=True, with_mask=True)

# clear mask of the molecule
moldata.set_mols_mask(None)

print_feature_extractor()

fs_atom = moldata.extract_features(HeavyAtomExtractor(moldata))
fs_atom.print_features()

# print information on the 10-th molecule in moldata
moldata.print_mol_info_by_index(10)

# extract the number of rings in molecules
fs_ring = moldata.extract_features(RingExtractor(moldata))
fs_ring.print_features()

# print information on the 10-th molecule in moldata
moldata.print_mol_info_by_index(10)

# extract the number of aromatic rings in molecules
fs_aring = moldata.extract_features(AromaticRingExtractor(moldata))
fs_aring.print_features()

# print information on the 10-th molecule in moldata
moldata.print_mol_info_by_index(10)

# extract fingerprint structure of radus 1
fs_fp_structure1 = moldata.extract_features(FingerPrintStructureExtractor(moldata, radius=1))
fs_fp_structure1.print_features()

# extract fingerprint structure of radus 2
fs_fp_structure2 = moldata.extract_features(FingerPrintStructureExtractor(moldata, radius=2))
# fs_fp_structure2.print_features() # too many for print all

# extract the total number of atoms from heavy atom count
fs_atom_sum = moldata.extract_features(FeatureSumOperator(moldata, fs_atom))
fs_atom_sum.print_features()

# check the result of sum operation
print(moldata.get_dataframe(features=[fs_atom_sum, fs_atom])[:5])

moldata.print_features()

fs_fp_all = moldata.merge_features([fs_fp_structure1.id])
features_fp = moldata.merge_features([fs_atom.id, fs_ring.id, fs_aring.id, fs_fp_structure1.id])
features_fp.print_features()
moldata.print_merged_features()

