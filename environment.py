import json
import copy
import re

from molgen import react
from chemprop.predict_one import predict_one
from utils import compute_molecular_mass


class Y6Environment:
    def __init__(self, reward_tp):
        self.inert_atoms = ['He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn']
        self.inert_pair_tuple_char = {
            'He': 'a',
            'Ne': 'b',
            'Ar': 'c',
            'Kr': 'd',
            'Xe': 'e',
            'Rn': 'f'
        }
        self.empty_state = {
                    'smiles': '',
                    'label': '',
                    'group': '',
                    'blocks': [{'smiles': ''}]
                    }
        self.reward_tp = reward_tp

    def get_side_chains_end_groups(self, json_path):
        f = json.load(open(json_path))
        side_chains = []
        end_groups = []

        for mol in f['molecules']:
            if mol['group'] == 'side_chain':
                side_chains.append(mol)
            elif mol['group'] == 'end_group':
                end_groups.append(mol)
        return side_chains, end_groups

    def find_lowest_inert_atom(self, str):
        for atom in self.inert_atoms:
            if '[' + atom + ']' in str:
                return atom

    def get_string_actions(self, tp):
        new_actions = []
        if tp == 'pos0':
            new_actions = [
                lambda str : str.replace('<pos0>', 's'),
                lambda str : str.replace('<pos0>', 'o'),
                lambda str : str.replace('<pos0>', 'c(C)c(C)'),
                lambda str : str.replace('<pos0>', 'cc'),
                lambda str : str.replace('n<pos0>n', 'cccc'),
                lambda str : str.replace('n<pos0>n', 'c(F)c(F)c(F)c(F)'),
                lambda str : str.replace('5n<pos0>nc5', '(F)c(F)'),
                lambda str : str.replace('<pos0>', 'n([Xe])')
            ]
        elif tp == 'pos1':
            new_actions = [
                lambda str : str.replace('<pos1>', 'n([Ar])'),
                lambda str : str.replace('<pos1>', 's'),
                lambda str : str.replace('<pos1>', 'o')
            ]
        elif tp == 'pos2':
            new_actions = [
                lambda str : str.replace('<pos2>', 'n([Kr])'),
                lambda str : str.replace('<pos2>', 's'),
                lambda str : str.replace('<pos2>', 'o')
            ]
        elif tp == 'pos3':
            new_actions = [
                lambda str : str.replace('<pos3>', 'n([Rn])'),
                lambda str : str.replace('<pos3>', 's'),
                lambda str : str.replace('<pos3>', 'o')
            ]
        return new_actions

    def get_next_actions_opd(self, state, side_chains, end_groups):
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
        elif ('[He]' in state['blocks'][0]['smiles']):
            new_actions = end_groups
        elif any(inert in state['blocks'][0]['smiles'] for inert in ['[Ne]', '[Ar]', '[Kr]', '[Xe]', '[Rn]']):
            new_actions = side_chains
        return new_actions

    def check_terminal(self, state):
        if 'blocks' in state and len(state['blocks']) == 0:
            return 1
        return 0

    def get_reward(self, smiles):
        if self.reward_tp == 'mass':
            return compute_molecular_mass(smiles), 0.0
        elif self.reward_tp == 'bandgap':
            prop, uncertainty = predict_one('models/weights_lite', [[smiles]])
            return -1 * prop[0][0], uncertainty[0]

    def write_to_tensorboard(self, writer, num, **kwargs):
        for key, metric in kwargs.items():
            writer.add_scalar(key, metric, num)
 
    def propagate_state(self, state, action):
        next_state = copy.deepcopy(self.empty_state)
        if callable(action) and action.__name__ == "<lambda>": # if it is a string manipulation action
            next_state['blocks'][0]['smiles'] = action(state['blocks'][0]['smiles'])

            cleaned_smiles = next_state['blocks'][0]['smiles']
            for inert_atom in self.inert_pair_tuple_char.keys():
                handle_finder = re.compile("\[(" + inert_atom + ")\]")
                cleaned_smiles = handle_finder.sub('[H]', cleaned_smiles)

            next_state['smiles'] = cleaned_smiles
        elif type(action) == dict: # if it is a react-sym action
            lowest_inert_atom = self.find_lowest_inert_atom(state['blocks'][0]['smiles'])

            if lowest_inert_atom != 'He': # it requires a sidechain
                modified_action = copy.deepcopy(action)
                modified_action['blocks'][0]['smiles'] = modified_action['blocks'][0]['smiles'].replace('inert', lowest_inert_atom)
            else:
                modified_action = action

            pair_tuple = (self.inert_pair_tuple_char[lowest_inert_atom], self.inert_pair_tuple_char[lowest_inert_atom])
            next_state = react.run('opd', core=state, functional_group=modified_action, reactive_pos=0, pair_tuple=pair_tuple)[0]
        return next_state
