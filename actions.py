import re
import random
import copy
from molgen import react
from utils import find_isotope_mass_from_string


class BaseAction:
    def __init__(self):
        pass

    def get_identifier(self):
        # return string identifier such as smiles for the action
        # this string will be stored in the sequence of actions when the molecule is saved
        pass

    def __call__(self):
        pass

    def cleanup(self, smi):
        handle_finder = re.compile("\[(" + '\d+He' + ")\]")
        return handle_finder.sub('[H]', smi)

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
        next_state = copy.deepcopy(state)
 
        smiles = state['smiles']
        new_smiles = callable_action(smiles)
        next_state['smiles'] = new_smiles

        action_group = 'core'

        return next_state, action_group

class DictAction(BaseAction):
    def __init__(self, action_dict):
        BaseAction.__init__(self)
        self.action_dict = action_dict

    def get_identifier(self):
        identifier_dict = copy.deepcopy(self.action_dict)
        identifier_dict['identifier'] = self.cleanup(self.action_dict['smiles'])
        identifier_dict['key'] = self.action_dict['group']
        return identifier_dict

    def __call__(self, state, pos1, pos2=100):
        next_state = react.run(state['smiles'], self.action_dict['smiles'], pos1, pos2)
        action_group = self.action_dict['group']

        return next_state, action_group
