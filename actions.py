import re
import random
import copy
from molgen import react
from utils import find_isotope_mass_from_string


class BaseAction:
    def __init__(self):
        self.empty_state = {
            'smiles': '',
            'label': '',
            'fragments': {'pos0': {}, 'pos1': {}, 'pos2': {}, 'pos3': {}, 'pi_bridge_1': {}, 'pi_bridge_2': {}, 'end_group': {}, 'side_chain': {}},
            'group': {'core': 0, 'end_group': 0, 'side_chain': 0, 'pi_bridge': 0, 'pi_bridge_terminate': 0},
            'blocks': [{'smiles': ''}],
        }
        self.inert_pair_tuple_char = {
            'He': 'a',
            'Ne': 'b',
            'Ar': 'c',
            'Kr': 'd',
            'Xe': 'e',
            'Rn': 'f'
        }
        self.inert_atoms = ['He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn']

    def find_lowest_mass(self, str):
        return min(find_isotope_mass_from_string(str))

    def get_identifier(self):
        # return string identifier such as smiles for the action
        # this string will be stored in the sequence of actions when the molecule is saved
        pass

    def __call__(self):
        pass


class StringAction(BaseAction):
    def __init__(self, keyword, replaced_text, asymmetric=False):
        BaseAction.__init__(self)
        self.keyword = keyword
        self.replaced_text = replaced_text
        self.asymmetric = asymmetric

    def get_identifier(self):
        res = re.search(r'<(.*)>', self.keyword).group(1)
        return {
            'identifier': self.replaced_text,
            'key': res,
            'position': res,
            'keyword': self.keyword,
            'replaced': self.replaced_text
        }
    
    def __call__(self, state):
        if not self.asymmetric:
            callable_action = lambda str : str.replace(self.keyword, self.replaced_text)
        else:
            callable_action = lambda str: (str.replace(self.keyword, self.replaced_text, 1)).replace(self.keyword, self.replaced_text[::-1], 1)
        next_state = copy.deepcopy(self.empty_state)
 
        smiles = state['blocks'][0]['smiles']
        new_smiles = callable_action(smiles)
        next_state['blocks'][0]['smiles'] = new_smiles

        cleaned_smiles = next_state['blocks'][0]['smiles'] 
        for inert_atom in self.inert_pair_tuple_char.keys():
            handle_finder = re.compile("\[(" + inert_atom + ")\]")
            cleaned_smiles = handle_finder.sub('[H]', cleaned_smiles)
        next_state['smiles'] = cleaned_smiles

        action_group = 'core'

        return next_state, action_group

class DictAction(BaseAction):
    def __init__(self, action_dict):
        BaseAction.__init__(self)
        self.action_dict = action_dict

    def get_identifier(self):
        identifier_dict = copy.deepcopy(self.action_dict)
        identifier_dict['identifier'] = self.action_dict['smiles']
        identifier_dict['key'] = self.action_dict['group']
        return identifier_dict

    def __call__(self, state, pos1, pos2=100):
        next_state = react.run(state['smiles'], self.action_dict['smiles'], pos1, pos2)
        action_group = self.action_dict['group']

        return next_state, action_group
