import os, sys

sys.path.insert(0, os.path.join('train_chemprop', 'chemprop'))

import json
import copy
import re
import numpy as np

from molgen import react
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import chemprop
from chemprop_inference import predict
from utils import compute_molecular_mass, get_identity_reward
from actions import StringAction, DictAction

# Function to generate Morgan fingerprints for a list of SMILES strings
def generate_morgan_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        fingerprint_array = np.array(fingerprint, dtype=int)
        return fingerprint_array
    else:
        return None

# Function to compute Tanimoto similarity between two fingerprints
def compute_tanimoto_similarity(fp1, fp2):
    intersection = np.sum(np.logical_and(fp1, fp2))
    union = np.sum(np.logical_or(fp1, fp2))
    similarity = intersection / union if union > 0 else 0.0
    return similarity

class BaseEnvironment:
    def __init__(self, reward_tp, output_dir, reduction):

        # define self.root_state in constructor of inheriting class
        self.root_state = {}
        self.empty_state = {
                    'smiles': '',
                    'label': '',
                    'fragments': {'pos0': "", 'pos1': "", 'pos2': "", 'pos3': "", 'pi_bridge_1': "", 'pi_bridge_2': "", 'end_group': "", 'side_chain': ""},
                    'group': {'core': 0, 'end_group': 0, 'side_chain': 0, 'pi_bridge': 0, 'pi_bridge_terminate': 0},
                    'blocks': [{'smiles': ''}],
                    }
        self.reward_tp = reward_tp
        self.output_dir = output_dir
        self.pi_bridge_ctr = 0
        self.get_fragments('fragments/core-fxn-y6-methyls.json')
        arguments = [
            '--test_path', '/dev/null',
            '--preds_path', '/dev/null',
            '--checkpoint_dir', 'train_chemprop/chemprop_weights'
        ]
        self.args = chemprop.args.PredictArgs().parse_args(arguments)
        self.model_objects = chemprop.train.load_model(args=self.args)
        self.reduction = reduction

    def get_root_state(self):
        return self.root_state

    def reset(self):
        self.pi_bridge_ctr = 0

    def get_string_actions(self, tp):
        new_actions = []
        if tp == 'pos0':
            new_actions = [
                StringAction('<pos0>', 's'),
                StringAction('<pos0>', 'o'),
                StringAction('<pos0>', 'c(C)c(C)'),
                StringAction('<pos0>', 'cc'),
                StringAction('n<pos0>n', 'cccc'),
                StringAction('5n<pos0>nc5', '(F)c(F)'),
                StringAction('<pos0>', 'n([Xe])')
            ]
        elif tp == 'pos1':
            new_actions = [
                StringAction('<pos1>', 'nc', asymmetric=True),
                StringAction('<pos1>', 'cn', asymmetric=True),
                StringAction('<pos1>', 'cc'),
                StringAction('<pos1>', 'n([Ar])'),
                StringAction('<pos1>', 's'),
                StringAction('<pos1>', 'o')
            ]
        elif tp == 'pos2':
            new_actions = [
                StringAction('<pos2>', 'nc', asymmetric=True),
                StringAction('<pos1>', 'cn', asymmetric=True),
                StringAction('<pos2>', 'cc'),
                StringAction('<pos2>', 'n([Kr])'),
                StringAction('<pos2>', 's'),
                StringAction('<pos2>', 'o')
            ]
        elif tp == 'pos3':
            new_actions = [
                StringAction('<pos3>', 'nc', asymmetric=True),
                StringAction('<pos1>', 'cn', asymmetric=True),
                StringAction('<pos3>', 'cc'),
                StringAction('<pos3>', 'n([Rn])'),
                StringAction('<pos3>', 's'),
                StringAction('<pos3>', 'o')
            ]
        return new_actions

    def check_terminal(self, state):
        if 'blocks' in state and len(state['blocks']) == 0:
            return 1
        return 0

    def get_reward(self, smiles):
        if self.reward_tp == 'mass':
            return compute_molecular_mass(smiles), 0, 0.0
        elif self.reward_tp == 'bandgap':
            # prop, uncertainty = predict_one('models/patent_mcts_checkpoints', [[smiles]])
            arguments = [
                '--test_path', '/dev/null',
                '--preds_path', '/dev/null',
                '--uncertainty_method', 'ensemble',
                '--checkpoint_dir', 'train_chemprop/chemprop_weights'
            ]
            args = chemprop.args.PredictArgs().parse_args(arguments)
            model_objects = chemprop.train.load_model(args=args)
            smiles = [[smiles]]
            preds = chemprop.train.make_predictions(args=args, smiles=smiles, model_objects=model_objects, return_uncertainty=True)
            return -1 * preds[0][0][1], 0, preds[1][0][1]
        elif self.reward_tp == 'tanimoto_bandgap':
            arguments = [
                '--test_path', '/dev/null',
                '--preds_path', '/dev/null',
                '--checkpoint_dir', 'train_chemprop/chemprop_weights'
            ]

            fp = generate_morgan_fingerprint(smiles)
            smiles = [[smiles]]
            preds = chemprop.train.make_predictions(args=self.args, smiles=smiles, model_objects=self.model_objects) #return_uncertainty=True)

            if os.path.exists(os.path.join(self.output_dir, 'fingerprints.npy')):
                saved_fps = np.load(os.path.join(self.output_dir, 'fingerprints.npy'))
                max_score = max([compute_tanimoto_similarity(saved_fp, fp) for saved_fp in saved_fps])
                # similarity_reward = 0 if max_score >= 0.378 else 1
                similarity_reward = max_score
            else:
                similarity_reward = get_identity_reward(reduction=self.reduction)
            return preds[0][1], similarity_reward, 0

    def write_to_tensorboard(self, writer, num, **kwargs):
        for key, metric in kwargs.items():
            writer.add_scalar(key, metric, num)

class Y6Environment(BaseEnvironment):
    def __init__(self, reward_tp, output_dir, reduction):
        BaseEnvironment.__init__(self, reward_tp, output_dir, reduction)
        self.root_state = {
            'smiles': "c1cc2<pos2>c3c4c5n<pos0>nc5c6c7<pos2>c8cc<pos3>c8c7<pos1>c6c4<pos1>c3c2<pos3>1",
            'label': 'opd',
            'fragments': {'pos0': "", 'pos1': "", 'pos2': "", 'pos3': "", 'pi_bridge_1': "", 'pi_bridge_2': "", 'end_group': "", 'side_chain': ""},
            'group': {'core': 0, 'end_group': 0, 'side_chain': 0, 'pi_bridge': 0, 'pi_bridge_terminate': 0},
            'blocks': [{'smiles': 'c1([He])c([Ne])c2<pos2>c3c4c5n<pos0>nc5c6c7<pos2>c8c([Ne])c([He])<pos3>c8c7<pos1>c6c4<pos1>c3c2<pos3>1'}]
        }

    def get_fragments(self, json_path):
        f = json.load(open(json_path))
        self.cores = []
        self.side_chains = []
        self.end_groups = []
        self.pi_bridges = []

        for mol in f['molecules']:
            if mol['group'] == 'side_chain':
                self.side_chains.append(DictAction(mol))
            elif mol['group'] == 'end_group':
                self.end_groups.append(DictAction(mol))
            elif mol['group'] == 'pi_bridge':
                self.pi_bridges.append(DictAction(mol))

    def get_next_actions_opd(self, state):
        if self.check_terminal(state):
            new_actions = []
        elif ('pos0' in state['blocks'][0]['smiles']):
            new_actions = self.get_string_actions('pos0')
        elif ('pos1' in state['blocks'][0]['smiles']):
            new_actions = self.get_string_actions('pos1')
        elif ('pos2' in state['blocks'][0]['smiles']):
            new_actions = self.get_string_actions('pos2')
        elif ('pos3' in state['blocks'][0]['smiles']):
            new_actions = self.get_string_actions('pos3')
        elif ('[He]' in state['blocks'][0]['smiles']) and (state['group']['pi_bridge'] < 2) and (state['group']['pi_bridge_terminate'] == 0):
            new_actions = self.pi_bridges + [DictAction({})] # None is empty action i.e. no bridge. If this action is chosen, proceed to end groups as next action
        elif ('[He]' in state['blocks'][0]['smiles']):
            new_actions = self.end_groups
        elif any(inert in state['blocks'][0]['smiles'] for inert in ['[Ne]', '[Ar]', '[Kr]', '[Xe]', '[Rn]']):
            new_actions = self.side_chains
        return new_actions

    def propagate_state(self, state, action):
        next_state, action_group = action(state)
        return next_state, action_group
