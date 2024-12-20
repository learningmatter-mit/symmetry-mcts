import re
import random
import copy
from rdkit import Chem
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
        handle_finder = re.compile("\[(" + "\d+He" + ")\]")
        cleaned_smi = handle_finder.sub("[H]", smi)
        cleaned_smi = Chem.MolToSmiles(Chem.RemoveHs(Chem.MolFromSmiles(cleaned_smi)))
        return cleaned_smi


class StringAction(BaseAction):
    def __init__(self, keyword, replaced_text, asymmetric=False):
        BaseAction.__init__(self)
        self.keyword = keyword
        self.replaced_text = replaced_text
        self.asymmetric = asymmetric

    def get_identifier(self):
        res = re.search(r"<(.*)>", self.keyword).group(1)
        return {
            "identifier": self.replaced_text,
            "key": res,
            "position": res,
            "keyword": self.keyword,
            "replaced": self.replaced_text,
        }

    def __call__(self, state, **kwargs):
        if not self.asymmetric:
            callable_action = lambda str: str.replace(self.keyword, self.replaced_text)
        else:
            callable_action = lambda str: (
                str.replace(self.keyword, self.replaced_text, 1)
            ).replace(self.keyword, self.replaced_text[::-1], 1)
        next_state = copy.deepcopy(state)

        smiles = state["smiles"]
        new_smiles = callable_action(smiles)
        next_state["smiles"] = new_smiles

        next_state["fragments"][self.get_identifier()["key"]].append(self.replaced_text)

        return next_state


class ClusterAction(BaseAction):
    def __init__(self, action_dict):
        BaseAction.__init__(self)
        self.action_dict = action_dict

    def get_identifier(self):
        identifier_dict = copy.deepcopy(self.action_dict)
        identifier_dict["identifier"] = self.action_dict["index"]
        identifier_dict["key"] = self.action_dict["group"]
        return identifier_dict

    def __call__(self, state, **kwargs):
        action_group = self.action_dict["group"]
        next_state = copy.deepcopy(state)
        if action_group == "terminate_pi_bridge":
            next_state["fragments"]["terminate_pi_bridge"] = 1
        else:
            next_state["fragments"][action_group].append(
                self.get_identifier()["identifier"]
            )
        return next_state


class DictAction(BaseAction):
    def __init__(self, action_dict):
        BaseAction.__init__(self)
        self.action_dict = action_dict
        if not self.is_valid():
            raise Exception(
                "Invalid smiles in action dict: ", self.action_dict["smiles"]
            )

    def is_valid(self):
        try:
            smi = Chem.MolFromSmiles(self.action_dict["smiles"])
            if smi != None:
                return True
            else:
                return False
        except:
            return False

    def get_identifier(self):
        identifier_dict = copy.deepcopy(self.action_dict)
        identifier_dict["identifier"] = self.cleanup(self.action_dict["smiles"])
        identifier_dict["key"] = self.action_dict["group"]
        return identifier_dict

    def __call__(self, state, **kwargs):
        action_group = self.action_dict["group"]
        next_state = copy.deepcopy(state)

        if action_group == "terminate_pi_bridge":
            next_state["fragments"]["terminate_pi_bridge"] = 1
            return next_state

        # State is root state and you choose first action
        if action_group == "core":
            next_state["smiles"] = self.action_dict["smiles"]
            next_state["fragments"][action_group].append(
                self.get_identifier()["identifier"]
            )
            return next_state

        pos1 = kwargs["pos1"]
        pos2 = kwargs["pos2"]

        new_smiles = react.run(state["smiles"], self.action_dict["smiles"], pos1, pos2)

        next_state = copy.deepcopy(state)
        next_state["smiles"] = new_smiles
        next_state["fragments"][action_group].append(
            self.get_identifier()["identifier"]
        )

        return next_state
