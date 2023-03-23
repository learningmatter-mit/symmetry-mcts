from rdkit.Chem import MolFromSmarts
#from molgen.singlereactant_rxnenv import SingleReactantRxnEnv
#from molgen.multismarts_rxnenv import MultiSmartsRxnEnv
#from molgen.rdkit_grafting_reaction import run_grafting_reactions
import collections
import datetime
import itertools
import time
import numpy
import logging

from rdkit.Chem.AllChem import CalcExactMolWt, MolFromSmiles, MolToSmiles
from molgen.factory import factory

logger = logging.getLogger()


def run(recipe_name, **kwargs):
    molecule = factory.create(recipe_name, **kwargs)
    products = molecule.react()
    return products

def cleanup(state):
    new_state = state.copy()
    state['blocks'] = []
    return new_state