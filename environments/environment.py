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
)
from environments.actions import StringAction, DictAction, ClusterAction


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
        if self.reward_tp == "mass":
            return compute_molecular_mass(smiles), 0, 0.0
        elif self.reward_tp == "bandgap":
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
            "train_chemprop/chemprop_weights",
        ]
        self.args = chemprop.args.PredictArgs().parse_args(arguments)
        self.model_objects = chemprop.train.load_model(args=self.args)

    def reset(self):
        self.pi_bridge_ctr = 0

    def get_fragments(self, json_path):
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
                StringAction("<pos2>", "cn", asymmetric=True),
                StringAction("<pos3>", "cc"),
                StringAction("<pos3>", "n([105He])"),
                StringAction("<pos3>", "s"),
                StringAction("<pos3>", "o"),
            ]
        return new_actions

    def get_next_actions(self, state):
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
        pos1 = min(find_isotope_mass_from_string(state["smiles"]))
        next_state = action(state, pos1=pos1, pos2=100)
        return next_state


class PatentEnvironment(BaseEnvironment):
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

    def check_terminal(self, state):
        if state["smiles"] == "":
            return 0
        if ("He" not in state["smiles"]) or compute_molecular_mass(
            self.fill_inert_positions(state["smiles"])
        ) >= 1500:
            return 1
        return 0

    def reset(self):
        # Nothing to reset
        pass

    def get_fragments(self, cores_path, bridges_path, end_groups_path):
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
        return self.fill_inert_positions(smiles)

    def get_next_actions(self, state):
        # If terminal state, then return empty action set
        if self.check_terminal(state):
            return []

        elif state["next_action"].startswith("cluster"):
            return self.clusters[state["next_action"]]
        else:
            try:
                self.cluster_to_frag[state["next_action"]][
                    state["fragments"]["cluster_" + state["next_action"]][-1]
                ]
            except:
                import pdb

                pdb.set_trace()
            return self.cluster_to_frag[state["next_action"]][
                state["fragments"]["cluster_" + state["next_action"]][-1]
            ]

    def process_next_state(self, next_state, next_action):
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
        for isotope_num in list(set(find_isotope_mass_from_string(smi))):
            smi = smi.replace(str(isotope_num) + "He", "H")
        return Chem.MolToSmiles(Chem.RemoveHs(Chem.MolFromSmiles(smi)))

    def propagate_state(self, state, action):
        if action.action_dict["group"] == "core" or isinstance(action, ClusterAction):
            next_state = action(state)
        else:
            try:
                pos1 = random.choice(
                    list(set(find_isotope_mass_from_string(state["smiles"])))
                )
            except:
                import pdb

                pdb.set_trace()

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
