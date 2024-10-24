import os, sys

sys.path.insert(0, os.path.join("train_chemprop", "chemprop"))

import json
import copy
import random
import re
import numpy as np
import pandas as pd

from molgen import react
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import chemprop
from chemprop_inference import predict
from utils import (
    compute_molecular_mass,
    get_identity_reward,
    find_isotope_mass_from_string,
    check_smiles_validity,
)
from environments.actions import StringAction, DictAction, ClusterAction


# Function to generate Morgan fingerprints for a list of SMILES strings
def generate_morgan_fingerprint(smiles, radius=2, n_bits=2048):
    """
    Generates a Morgan fingerprint for a given SMILES string.

    Parameters:
    smiles (str): The SMILES representation of the molecule.
    radius (int, optional): The radius of the Morgan fingerprint. Default is 2.
    n_bits (int, optional): The number of bits in the fingerprint. Default is 2048.

    Returns:
    np.ndarray or None: A numpy array representing the Morgan fingerprint if the SMILES string is valid, 
                        otherwise None.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        fingerprint_array = np.array(fingerprint, dtype=int)
        return fingerprint_array
    else:
        return None


# Function to compute Tanimoto similarity between two fingerprints
def compute_tanimoto_similarity(fp1, fp2):
    """
    Compute the Tanimoto similarity between two binary fingerprints.

    The Tanimoto similarity is defined as the size of the intersection divided by the size of the union of the two sets.

    Parameters:
    fp1 (numpy.ndarray): The first binary fingerprint.
    fp2 (numpy.ndarray): The second binary fingerprint.

    Returns:
    float: The Tanimoto similarity between the two fingerprints. The value ranges from 0.0 to 1.0, where 1.0 indicates identical fingerprints and 0.0 indicates no similarity.
    """
    intersection = np.sum(np.logical_and(fp1, fp2))
    union = np.sum(np.logical_or(fp1, fp2))
    similarity = intersection / union if union > 0 else 0.0
    return similarity


class BaseEnvironment:
    """
    BaseEnvironment is an abstract base class for defining environments in the molMCTS framework. 
    It provides a structure for initializing the environment, resetting it, retrieving possible actions, 
    and calculating rewards based on different criteria.

    Attributes:
        root_state (dict): The initial state of the environment.
        empty_state (dict): A placeholder for an empty state.
        reward_tp (str): The type of reward to be calculated ("bandgap" or "tanimoto_bandgap").
        output_dir (str): Directory where output files are stored.
        reduction (str): Reduction method used for calculating identity reward.

    Methods:
        __init__(reward_tp, output_dir, reduction):
            Initializes the environment with the specified reward type, output directory, and reduction method.
        
        reset():
            Resets the environment to its initial state. Must be implemented in the inheriting class.
        
        get_next_actions(state):
            Retrieves the possible next actions from the given state. Must be implemented in the inheriting class.
        
        get_root_state():
            Returns the root state of the environment.
        
        check_terminal(state):
            Checks if the given state is a terminal state. Returns 1 if terminal, otherwise 0.
        
        get_reward(smiles):
            Calculates the reward for the given SMILES string based on the specified reward type.
        
        write_to_tensorboard(writer, num, **kwargs):
            Writes metrics to TensorBoard for visualization.
    """
    def __init__(self, reward_tp, output_dir, reduction):

        # define self.root_state and self.empty_state in constructor of inheriting class
        self.root_state = {}
        self.empty_state = {}

        self.reward_tp = reward_tp
        self.output_dir = output_dir
        self.reduction = reduction

    def reset(self):
        # define in inherited class
        pass

    def get_next_actions(self, state):
        # define in inherited class
        pass

    def get_root_state(self):
        return self.root_state

    def check_terminal(self, state):
        if "He" not in state["smiles"] and state != self.root_state:
            return 1
        return 0

    def get_reward(self, smiles):
        if self.reward_tp == "bandgap":
            smiles = [[smiles]]
            preds = chemprop.train.make_predictions(
                args=self.args,
                smiles=smiles,
                model_objects=self.model_objects,
                return_uncertainty=True,
            )
            return -1 * preds[0][0][1], 0, preds[1][0][1]
        elif self.reward_tp == "tanimoto_bandgap":
            fp = generate_morgan_fingerprint(smiles)
            smiles = [[smiles]]
            preds = chemprop.train.make_predictions(
                args=self.args, smiles=smiles, model_objects=self.model_objects
            )  # return_uncertainty=True)

            if os.path.exists(os.path.join(self.output_dir, "fingerprints.npy")):
                saved_fps = np.load(os.path.join(self.output_dir, "fingerprints.npy"))
                max_score = max(
                    [
                        compute_tanimoto_similarity(saved_fp, fp)
                        for saved_fp in saved_fps
                    ]
                )
                similarity_reward = max_score
            else:
                similarity_reward = get_identity_reward(reduction=self.reduction)
            return preds[0][1], similarity_reward, 0

    def write_to_tensorboard(self, writer, num, **kwargs):
        for key, metric in kwargs.items():
            writer.add_scalar(key, metric, num)


class Y6Environment(BaseEnvironment):
    """
    Y6Environment is a specialized environment for molecular generation and manipulation.

    Attributes:
        root_state (dict): The initial state of the environment.
        empty_state (dict): A deepcopy of the root_state with an empty "smiles" string.
        pi_bridge_ctr (int): Counter for pi bridges.
        args (chemprop.args.PredictArgs): Arguments for the chemprop model.
        model_objects (list): Loaded chemprop model objects.
        cores (list): List of core fragments.
        side_chains (list): List of side chain fragments.
        end_groups (list): List of end group fragments.
        pi_bridges (list): List of pi bridge fragments.

    Methods:
        __init__(reward_tp, output_dir, reduction):
            Initializes the Y6Environment with the given parameters.
        
        reset():
            Resets the pi_bridge_ctr to 0.
        
        get_fragments(json_path):
            Loads fragments from a JSON file and categorizes them into cores, side chains, end groups, and pi bridges.
        
        get_string_actions(tp):
            Returns a list of StringAction objects based on the type of position (tp).
        
        get_next_actions(state):
            Determines the next possible actions based on the current state.
        
        process_next_state(next_state, next_action):
            Updates the next state based on the next action taken.
        
        propagate_state(state, action):
            Propagates the state by applying the given action.
    """
    def __init__(self, reward_tp, output_dir, reduction):
        BaseEnvironment.__init__(self, reward_tp, output_dir, reduction)
        self.root_state = {
            "smiles": "c1([100He])c([101He])c2<pos2>c3c4c5n<pos0>nc5c6c7<pos2>c8c([101He])c([100He])<pos3>c8c7<pos1>c6c4<pos1>c3c2<pos3>1",
            "fragments": {
                "pos0": [],
                "pos1": [],
                "pos2": [],
                "pos3": [],
                "pi_bridge": [],
                "end_group": [],
                "side_chain": [],
                "terminate_pi_bridge": 0,
            },
            "next_action": "pos0",
        }
        self.empty_state = copy.deepcopy(self.root_state)
        self.empty_state["smiles"] = ""
        self.pi_bridge_ctr = 0
        self.get_fragments("fragments/core-fxn-y6-methyls.json")
        arguments = [
            "--test_path",
            "/dev/null",
            "--preds_path",
            "/dev/null",
            "--checkpoint_dir",
            "train_chemprop/chemprop_weights_y6",
        ]
        self.args = chemprop.args.PredictArgs().parse_args(arguments)
        self.model_objects = chemprop.train.load_model(args=self.args)

    def reset(self):
        self.pi_bridge_ctr = 0

    def get_fragments(self, json_path):
        """
        Parses a JSON file containing molecular fragments and categorizes them into cores, side chains, end groups, and pi bridges.

        Args:
            json_path (str): The file path to the JSON file containing molecular fragments.

        Attributes:
            cores (list): A list to store core fragments (currently not populated in this method).
            side_chains (list): A list to store side chain fragments.
            end_groups (list): A list to store end group fragments.
            pi_bridges (list): A list to store pi bridge fragments, including a terminating pi bridge fragment.

        The JSON file is expected to have the following structure:
        {
            "molecules": [
                {
                    "group": "side_chain" | "end_group" | "pi_bridge",
                    "smiles": "..."
                },
                ...
            ]
        }
        """
        f = json.load(open(json_path))
        self.cores = []
        self.side_chains = []
        self.end_groups = []
        self.pi_bridges = []

        for mol in f["molecules"]:
            if mol["group"] == "side_chain":
                self.side_chains.append(DictAction(mol))
            elif mol["group"] == "end_group":
                self.end_groups.append(DictAction(mol))
            elif mol["group"] == "pi_bridge":
                self.pi_bridges.append(DictAction(mol))
        self.pi_bridges.append(
            DictAction({"smiles": "", "group": "terminate_pi_bridge"})
        )

    def get_string_actions(self, tp):
        """
        Generate a list of StringAction objects based on the specified type.

        Args:
            tp (str): The type of position for which to generate actions. 
                      It can be one of the following values: "pos0", "pos1", "pos2", "pos3".

        Returns:
            list: A list of StringAction objects corresponding to the specified type.

        Raises:
            ValueError: If the provided type is not one of the expected values.
        """
        new_actions = []
        if tp == "pos0":
            new_actions = [
                StringAction("<pos0>", "s"),
                StringAction("<pos0>", "o"),
                StringAction("<pos0>", "c(C)c(C)"),
                StringAction("<pos0>", "cc"),
                StringAction("n<pos0>n", "cccc"),
                StringAction("5n<pos0>nc5", "(F)c(F)"),
                StringAction("<pos0>", "n([102He])"),
            ]
        elif tp == "pos1":
            new_actions = [
                StringAction("<pos1>", "nc", asymmetric=True),
                StringAction("<pos1>", "cn", asymmetric=True),
                StringAction("<pos1>", "cc"),
                StringAction("<pos1>", "n([103He])"),
                StringAction("<pos1>", "s"),
                StringAction("<pos1>", "o"),
            ]
        elif tp == "pos2":
            new_actions = [
                StringAction("<pos2>", "nc", asymmetric=True),
                StringAction("<pos2>", "cn", asymmetric=True),
                StringAction("<pos2>", "cc"),
                StringAction("<pos2>", "n([104He])"),
                StringAction("<pos2>", "s"),
                StringAction("<pos2>", "o"),
            ]
        elif tp == "pos3":
            new_actions = [
                StringAction("<pos3>", "nc", asymmetric=True),
                StringAction("<pos3>", "cn", asymmetric=True),
                StringAction("<pos3>", "cc"),
                StringAction("<pos3>", "n([105He])"),
                StringAction("<pos3>", "s"),
                StringAction("<pos3>", "o"),
            ]
        return new_actions

    def get_next_actions(self, state):
        """
        Determines the next possible actions based on the current state.

        Args:
            state (dict): The current state of the environment, which includes 
                          information about the next action to be taken.

        Returns:
            list: A list of possible next actions. If the state is terminal, 
                  an empty list is returned. Otherwise, the list of actions 
                  depends on the value of `state["next_action"]`:
                  - If it contains "pos", the result of `get_string_actions` is returned.
                  - If it is "pi_bridge", `pi_bridges` is returned.
                  - If it is "end_group", `end_groups` is returned.
                  - If it is "side_chain", `side_chains` is returned.
        """
        if self.check_terminal(state):
            return []
        elif "pos" in state["next_action"]:
            return self.get_string_actions(state["next_action"])
        elif state["next_action"] == "pi_bridge":
            return self.pi_bridges
        elif state["next_action"] == "end_group":
            return self.end_groups
        elif state["next_action"] == "side_chain":
            return self.side_chains

    def process_next_state(self, next_state, next_action):
        """
        Processes the next state based on the given action.

        Parameters:
        next_state (dict): The current state of the environment, which will be updated based on the action.
        next_action (Action): The action to be processed, which contains an identifier used to determine the next state.

        Returns:
        dict: The updated state after processing the action.
        """
        key = next_action.get_identifier()["key"]

        if key == "pos0":
            next_state["next_action"] = "pos1"
        elif key == "pos1":
            next_state["next_action"] = "pos2"
        elif key == "pos2":
            next_state["next_action"] = "pos3"
        elif key == "pos3":
            next_state["next_action"] = "pi_bridge"
        elif key == "pi_bridge":
            if (len(next_state["fragments"]["pi_bridge"]) < 2) and (
                next_state["fragments"]["terminate_pi_bridge"] == 0
            ):
                next_state["next_action"] = "pi_bridge"
            else:
                next_state["next_action"] = "end_group"
        elif key == "end_group":
            next_state["next_action"] = "side_chain"

        return next_state

    def propagate_state(self, state, action):
        """
        Propagates the given state by applying the specified action.

        Args:
            state (dict): A dictionary representing the current state, which must include a "smiles" key.
            action (callable): A function that takes the current state and positional arguments `pos1` and `pos2`, 
                               and returns the next state.

        Returns:
            dict: The next state after applying the action.
        """
        pos1 = min(find_isotope_mass_from_string(state["smiles"]))
        next_state = action(state, pos1=pos1, pos2=100)
        return next_state


class PatentEnvironment(BaseEnvironment):
    """
    PatentEnvironment is a specialized environment for handling molecular fragments and their assembly
    into larger structures. It extends the BaseEnvironment class and provides methods for managing
    molecular states, checking terminal conditions, and processing actions.

    Attributes:
        root_state (dict): The initial state of the environment.
        empty_state (dict): A deep copy of the root state with an empty "smiles" string.
        args (chemprop.args.PredictArgs): Arguments for the chemprop model.
        model_objects (list): Loaded chemprop model objects.
        cluster_to_frag (dict): Mapping of cluster numbers to lists of allowed fragments.
        clusters (dict): List of allowed cluster indices.

    Methods:
        __init__(reward_tp, output_dir, reduction):
            Initializes the PatentEnvironment with the given parameters.
        
        getnumheavyatoms(smiles):
            Returns the number of heavy atoms in the given SMILES string.
        
        check_terminal(state):
            Checks if the given state is terminal.
        
        reset():
            Resets the environment. Currently, this method does nothing.
        
        get_fragments(cores_path, bridges_path, end_groups_path):
            Loads fragment data from the given CSV files and maps cluster numbers to allowed fragments.
        
        postprocess_smiles(smiles):
            Fills inert positions in the given SMILES string.
        
        get_next_actions(state):
            Returns the next possible actions for the given state.
        
        process_next_state(next_state, next_action):
            Processes the next state based on the given action.
        
        fill_inert_positions(smi):
            Replaces inert positions in the given SMILES string with hydrogen atoms.
        
        propagate_state(state, action):
            Propagates the state based on the given action.
    """
    def __init__(self, reward_tp, output_dir, reduction):
        BaseEnvironment.__init__(self, reward_tp, output_dir, reduction)
        self.root_state = {
            "smiles": "",
            "fragments": {
                "core": [],
                "pi_bridge": [],
                "end_group": [],
                "cluster_core": [],
                "cluster_pi_bridge": [],
                "cluster_end_group": [],
                "terminate_pi_bridge": 0,
            },
            "next_action": "cluster_core",
        }
        self.empty_state = copy.deepcopy(self.root_state)
        self.empty_state["smiles"] = ""
        # self.get_fragments("fragments/fragments_patents_with_Si.csv")
        self.get_fragments(
            "fragments/cores_with_clusters.csv",
            "fragments/bridges_with_clusters.csv",
            "fragments/end_groups_with_clusters.csv",
        )
        arguments = [
            "--test_path",
            "/dev/null",
            "--preds_path",
            "/dev/null",
            "--checkpoint_dir",
            "train_chemprop/chemprop_weights_frag_decomp",
        ]
        self.args = chemprop.args.PredictArgs().parse_args(arguments)
        self.model_objects = chemprop.train.load_model(args=self.args)

    def getnumheavyatoms(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        return mol.GetNumHeavyAtoms()

    def check_terminal(self, state):
        """
        Check if the given state is terminal.

        A state is considered terminal if:
        1. The "smiles" string in the state is empty.
        2. The "smiles" string is not a valid SMILES string.
        3. The "smiles" string does not contain the element Helium ("He") or 
           the number of heavy atoms in the "smiles" string is 100 or more.

        Args:
            state (dict): A dictionary representing the state, which must contain a "smiles" key.

        Returns:
            int: Returns 0 if the state is not terminal, otherwise returns 1.
        """
        if state["smiles"] == "":
            return 0

        if not check_smiles_validity(state["smiles"]):
            return 1

        if ("He" not in state["smiles"]) or self.getnumheavyatoms(
            state["smiles"]
        ) >= 100:
            return 1

        return 0

    def reset(self):
        # Nothing to reset
        pass

    def get_fragments(self, cores_path, bridges_path, end_groups_path):
        """
        Reads fragment data from CSV files and maps cluster numbers to lists of allowed fragments.

        Args:
            cores_path (str): Path to the CSV file containing core fragments.
            bridges_path (str): Path to the CSV file containing bridge fragments.
            end_groups_path (str): Path to the CSV file containing end group fragments.

        Returns:
            None

        Side Effects:
            - Populates self.cluster_to_frag with a dictionary mapping fragment types to clusters and their corresponding fragments.
            - Populates self.clusters with a dictionary mapping fragment types to lists of allowed cluster indices.
        """
        df_cores = pd.read_csv(cores_path)
        df_bridges = pd.read_csv(bridges_path)
        df_end_groups = pd.read_csv(end_groups_path)

        dfs = {"core": df_cores, "pi_bridge": df_bridges, "end_group": df_end_groups}

        # Map cluster numbers to list of allowed fragments
        self.cluster_to_frag = {"core": {}, "pi_bridge": {}, "end_group": {}}
        for frag_tp in self.cluster_to_frag.keys():
            num_pos = dfs[frag_tp]["cluster"]
            for i, frag in enumerate(dfs[frag_tp]["fragments"]):
                cluster_idx = num_pos[i]
                if cluster_idx not in self.cluster_to_frag[frag_tp]:
                    self.cluster_to_frag[frag_tp][cluster_idx] = [
                        DictAction({"smiles": frag, "group": frag_tp})
                    ]
                else:
                    self.cluster_to_frag[frag_tp][cluster_idx].append(
                        DictAction({"smiles": frag, "group": frag_tp})
                    )

        # List of allowed cluster indices
        self.clusters = {}
        for frag_tp in self.cluster_to_frag.keys():
            if frag_tp == "pi_bridge":  # Add terminal action in pi_bridge case
                self.clusters["cluster_" + frag_tp] = [
                    ClusterAction({"index": i, "group": "cluster_" + frag_tp})
                    for i in range(100)
                ] + [ClusterAction({"index": -1, "group": "terminate_pi_bridge"})]
            self.clusters["cluster_" + frag_tp] = [
                ClusterAction({"index": i, "group": "cluster_" + frag_tp})
                for i in range(100)
            ]

    def postprocess_smiles(self, smiles):
        """
        Post-processes a given SMILES string by filling inert positions.

        Args:
            smiles (str): The SMILES string to be post-processed.

        Returns:
            str: The post-processed SMILES string with inert positions filled.
        """
        return self.fill_inert_positions(smiles)

    def get_next_actions(self, state):
        """
        Determines the next possible actions based on the current state.

        Args:
            state (dict): The current state of the environment. It should contain:
                - "next_action" (str): The next action to be taken.
                - "fragments" (dict): A dictionary of fragments with their respective clusters.

        Returns:
            list: A list of possible next actions. If the state is terminal, returns an empty list.
        """
        # If terminal state, then return empty action set
        if self.check_terminal(state):
            return []

        elif state["next_action"].startswith("cluster"):
            return self.clusters[state["next_action"]]
        else:
            return self.cluster_to_frag[state["next_action"]][
                state["fragments"]["cluster_" + state["next_action"]][-1]
            ]

    def process_next_state(self, next_state, next_action):
        """
        Processes the next state based on the given next action and updates the next action to be taken.

        Args:
            next_state (dict): The current state of the environment, which will be updated based on the next action.
            next_action (object): The action to be processed, which contains an identifier used to determine the next action.

        Returns:
            dict: The updated state with the next action to be taken.
        """
        key = next_action.get_identifier()["key"]

        # Tag state with identity of next action choice to make it easy
        # during action selection
        if key == "cluster_core":
            next_state["next_action"] = "core"

        elif key == "core":
            next_state["next_action"] = "cluster_pi_bridge"

        elif key == "cluster_pi_bridge":
            next_state["next_action"] = "pi_bridge"

        elif key == "terminate_pi_bridge":
            next_state["next_action"] = "cluster_end_group"

        elif key == "pi_bridge":
            if (len(next_state["fragments"]["pi_bridge"]) < 2) and (
                next_state["fragments"]["terminate_pi_bridge"] == 0
            ):
                next_state["next_action"] = "cluster_pi_bridge"
            else:
                next_state["next_action"] = "cluster_end_group"

        elif key == "cluster_end_group":
            next_state["next_action"] = "end_group"

        elif key == "end_group":
            if "He" in next_state["smiles"]:
                next_state["next_action"] = "cluster_end_group"

        return next_state

    def fill_inert_positions(self, smi):
        """
        Replaces isotopic helium atoms in a SMILES string with hydrogen atoms and returns the modified SMILES string.

        Args:
            smi (str): A SMILES string representing a molecule.

        Returns:
            str: A SMILES string with isotopic helium atoms replaced by hydrogen atoms and explicit hydrogens removed.
        """
        smi_edited = smi
        for isotope_num in list(set(find_isotope_mass_from_string(smi))):
            smi_edited = smi_edited.replace(str(isotope_num) + "He", "H")
        return Chem.MolToSmiles(Chem.RemoveHs(Chem.MolFromSmiles(smi_edited)))

    def propagate_state(self, state, action):
        """
        Propagates the given state based on the specified action.

        Parameters:
        state (dict): The current state represented as a dictionary, typically containing a "smiles" key.
        action (Action): The action to be applied to the state. This can be an instance of ClusterAction or other action types.

        Returns:
        dict: The next state after applying the action.

        Notes:
        - If the action belongs to the "core" group or is an instance of ClusterAction, it is directly applied to the state.
        - For other actions, a random position from the state's "smiles" string is chosen.
        - If the action belongs to the "pi_bridge" group, special handling is done to maintain symmetry of cores in pi-bridges.
        - The function ensures that the "smiles" string in the next state is correctly updated.
        """
        if action.action_dict["group"] == "core" or isinstance(action, ClusterAction):
            next_state = action(state)
        else:
            pos1 = random.choice(
                list(set(find_isotope_mass_from_string(state["smiles"])))
            )

            action_positions = list(
                set(find_isotope_mass_from_string(action.action_dict["smiles"]))
            )
            random_index = random.randint(0, len(action_positions) - 1)
            pos2 = action_positions[random_index]

            if action.action_dict["group"] == "pi_bridge":
                other_pos = action_positions[1 - random_index]

                # This line is necessary to maintain symmetry of cores in pi-bridges
                # First replace by some filler atomic num, and then later replace by pos1
                action_bridge = copy.deepcopy(action)
                action_bridge.action_dict["smiles"] = action_bridge.action_dict[
                    "smiles"
                ].replace(str(other_pos) + "He", str(1000) + "He")
                next_state = action_bridge(state, pos1=pos1, pos2=pos2)
                next_state["smiles"] = next_state["smiles"].replace(
                    str(1000) + "He", str(pos1) + "He"
                )
            else:
                next_state = action(state, pos1=pos1, pos2=pos2)

        return next_state
