#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import pandas as pd
import copy
import json
import argparse
import pickle

# from chemprop.predict_one import predict_one
from torch.utils.tensorboard import SummaryWriter
from utils import (
    create_dir,
    get_num_atoms,
    get_num_atoms_by_id,
    set_all_seeds,
    get_total_reward,
    check_smiles_validity,
)
from environments.factory import factory
from tree_node import Tree_node


class MCTS:
    """
    Monte Carlo Tree Search (MCTS) class for performing tree-based search algorithms.

    Attributes:
        C (float): Exploration parameter for the UCB algorithm.
        environment (Environment): The environment in which the MCTS operates.
        exploration (str): The exploration strategy to use ("UCB" or "random").
        num_sims (int): The number of simulations to run.
        reward_tp (str): The type of reward to use ("bandgap" or other).
        reduction (str): The reduction method to use for rewards ("sum" or other).
        root (Tree_node): The root node of the MCTS tree.
        stable_structures_dict (dict): Dictionary to store stable structures.
        stable_structures_props (dict): Dictionary to store properties of stable structures.
        stable_structures_action_history (dict): Dictionary to store action history of stable structures.

    Methods:
        __init__(self, C, environment, exploration="UCB", num_sims=5000, reward_tp="bandgap", reduction="sum"):
            Initializes the MCTS object with the given parameters.

        save_outputs(self, final_state, metrics, num):

        get_metrics(self, gap_reward, sim_reward, reward, uncertainty, smiles):

        traverse(self, node, num, **kwargs):

        expand(self, node, **kwargs):

        roll_out(self, node, **kwargs):

        backprop(self, node, rw):

        save_tree(self, filename):

        load_tree(self, filename):

        run_sim(self, num):

        run(self, load=False):
        """

    def __init__(
        self,
        C,
        environment,
        exploration="UCB",
        num_sims=5000,
        reward_tp="bandgap",
        reduction="sum",
    ):

        self.C = C
        self.root = Tree_node([], self.C, None, 0)
        self.environment = environment

        self.stable_structures_dict = {}

        self.stable_structures_props = {}
        self.stable_structures_action_history = {}

        self.num_sims = num_sims
        self.reward_tp = reward_tp
        self.exploration = exploration
        self.reduction = reduction

    def save_outputs(self, final_state, metrics, num):
        """
        Save the outputs of the MCTS process including the final state and metrics.

        Parameters:
        final_state (dict): A dictionary containing the final state information, including "smiles" and "fragments".
        metrics (dict): A dictionary containing various metrics to be saved.
        num (int): An integer used to determine when to save the outputs to files.

        This method updates the stable structures dictionary and action history with the provided final state and metrics.
        It periodically saves these updates to CSV and JSON files based on the value of `num`.

        The stable structures dictionary (`self.stable_structures_dict`) is updated with the "smiles" from the final state
        and the provided metrics. The action history (`self.stable_structures_action_history`) is updated with the "smiles"
        and "fragments" from the final state.

        If `num` is divisible by 1, the method saves the stable structures dictionary to a CSV file and the action history
        to a JSON file. The file paths are determined by `iter_dir` and `fname_params`.
        """
        if "smiles" not in self.stable_structures_dict:
            self.stable_structures_dict["smiles"] = [final_state["smiles"]]
        else:
            self.stable_structures_dict["smiles"].append(final_state["smiles"])

        if final_state["smiles"] not in self.stable_structures_action_history.keys():
            self.stable_structures_action_history[final_state["smiles"]] = final_state[
                "fragments"
            ]

        for key in metrics.keys():
            if key not in self.stable_structures_dict:
                self.stable_structures_dict[key] = [metrics[key]]
            else:
                self.stable_structures_dict[key].append(metrics[key])

        if num % 1 == 0:
            df_stable_structures = pd.DataFrame.from_dict(self.stable_structures_dict)
            df_stable_structures.to_csv(
                os.path.join(iter_dir, fname_params["molecules_fname"]), index=False
            )

            with open(
                os.path.join(iter_dir, fname_params["action_history_fname"]), "w"
            ) as f:
                json.dump(self.stable_structures_action_history, f)

    def get_metrics(self, gap_reward, sim_reward, reward, uncertainty, smiles):
        """
        Calculate and return a dictionary of metrics.

        Args:
            gap_reward (float): The reward based on the gap.
            sim_reward (float): The reward based on similarity.
            reward (float): The overall reward.
            uncertainty (float): The uncertainty measure.
            smiles (str): The SMILES representation of the molecule.

        Returns:
            dict: A dictionary containing the calculated metrics.
        """
        metrics = {
            "gap_reward": gap_reward,
            "sim_reward": sim_reward,
            "reward": reward,
            "C": self.C,
            "uncertainty": uncertainty,
        }
        return metrics

    def traverse(self, node, num, **kwargs):
        """
        Traverse the MCTS tree starting from the given node.

        This method recursively traverses the tree based on the exploration strategy.
        If the node has no children or is a terminal state, it returns the node.
        Otherwise, it selects the next node to traverse based on the exploration strategy.

        Parameters:
        node (Node): The current node in the MCTS tree.
        num (int): An identifier or counter used during traversal.
        **kwargs: Additional keyword arguments.

        Returns:
        Node: The next node in the MCTS tree based on the exploration strategy.
        """
        if (len(node.children) == 0) or (self.environment.check_terminal(node.state)):
            return node
        else:
            if self.exploration == "random":
                rand_index = np.random.randint(0, len(node.children))
                return self.traverse(node.children[rand_index], num)
            elif self.exploration == "UCB":
                max_next = -100000
                index_max = -1
                curr_ucbs = []
                for i, child in enumerate(node.children):
                    curr_ucb = child.get_UCB(self.exploration, self.C)
                    curr_ucbs.append(curr_ucb)
                    if curr_ucb > max_next:
                        max_next = curr_ucb
                        index_max = i
                return self.traverse(node.children[index_max], num)

    def expand(self, node, **kwargs):
        """
        Expands the given node by generating its children nodes based on possible next actions.

        Args:
            node (Tree_node): The current node to be expanded.
            **kwargs: Additional arguments (not used in this function).

        Returns:
            Tree_node: A randomly selected child node from the newly created children nodes.

        The function performs the following steps:
        1. Retrieves the current state from the given node.
        2. Obtains the possible next actions from the environment based on the current state.
        3. For each possible next action:
            a. Propagates the current state to the next state using the action.
            b. Processes the next state using the environment's processing method.
            c. Creates a new Tree_node with the next state, parent node, and terminal state check.
            d. Appends the new node to the list of children of the current node.
        4. Randomly selects one of the newly created child nodes and returns it.
        """
        curr_state = node.state
        next_actions = self.environment.get_next_actions(curr_state)

        next_nodes = []
        for na in next_actions:
            next_state = self.environment.propagate_state(curr_state, na)

            next_state = self.environment.process_next_state(next_state, na)
            new_node = Tree_node(
                next_state, self.C, node, self.environment.check_terminal(next_state)
            )
            next_nodes.append(new_node)
            node.children.append(new_node)

        move = np.random.randint(0, len(next_actions))
        return next_nodes[move]

    def roll_out(self, node, **kwargs):
        """
        Perform a rollout (simulation) from the given node until a terminal state is reached.

        Args:
            node (Node): The starting node for the rollout.
            **kwargs: Additional keyword arguments.

        Returns:
            State: The terminal state reached after the rollout.
        """
        state = copy.deepcopy(node.state)
        while not self.environment.check_terminal(state):
            next_actions = self.environment.get_next_actions(state)
            move = np.random.randint(0, len(next_actions))
            next_action = next_actions[move]
            next_state = self.environment.propagate_state(state, next_action)

            next_state = self.environment.process_next_state(next_state, next_action)
            state = next_state
        return state

    def backprop(self, node, rw):
        """
        Perform backpropagation in the MCTS tree.

        This method updates the visit count and total reward of the given node and 
        recursively updates its parent nodes.

        Args:
            node (Node): The current node to update.
            rw (float): The reward to propagate up the tree.
        """
        node.inc_n()
        node.inc_T(rw)
        if node.parent != None:
            self.backprop(node.parent, rw)

    def save_tree(self, filename):
        """
        Save the current MCTS tree to a file.

        Args:
            filename (str): The path to the file where the tree will be saved.
        """
        with open(filename, "wb") as f:
            pickle.dump(self.root, f)

    def load_tree(self, filename):
        """
        Load a tree structure from a file.

        Args:
            filename (str): The path to the file containing the serialized tree structure.

        Raises:
            FileNotFoundError: If the file does not exist.
            IOError: If there is an error reading the file.
            pickle.UnpicklingError: If there is an error unpickling the file content.

        """
        with open(filename, "rb") as f:
            self.root = pickle.load(f)

    def run_sim(self, num):
        """
        Executes a single simulation of the Monte Carlo Tree Search (MCTS) algorithm.

        Args:
            num (int): The current iteration number of the simulation.

        Returns:
            float: The reward obtained from the simulation.

        Workflow:
            1. Selection: Traverse the tree from the root node to a leaf node.
            2. Expansion: Expand the leaf node if it is not terminal.
            3. Simulation/Roll-out: Simulate the outcome from the leaf node to a final state.
            4. Reward Calculation: Calculate the rewards based on the final state.
            5. Backpropagation: Propagate the reward back through the tree.
            6. Output: Save the outputs and metrics of the simulation.

        Notes:
            - The function checks the validity of the SMILES string in the final state.
            - Rewards are calculated based on the environment's reward function.
            - Metrics are written to TensorBoard for visualization.
            - Outputs are saved for further analysis.
        """
        print("Iteration: ", num)
        # selection
        curr_node = self.root
        leaf_node = self.traverse(curr_node, num)

        # expansion and not check_terminal(leaf_node.state)
        if leaf_node.get_n() != 0 and not self.environment.check_terminal(
            leaf_node.state
        ):
            leaf_node = self.expand(leaf_node)

        # simulation/roll_out
        final_state = self.roll_out(leaf_node)

        if not check_smiles_validity(final_state["smiles"]):
            gap_reward, sim_reward, uncertainty = float("inf"), float("inf"), 0.0
        else:
            if hasattr(self.environment, "postprocess_smiles"):
                final_state["smiles"] = self.environment.postprocess_smiles(
                    final_state["smiles"]
                )
            gap_reward, sim_reward, uncertainty = self.environment.get_reward(
                final_state["smiles"]
            )

        if os.path.exists(os.path.join(args.output_dir, "fingerprints.npy")):
            reward = get_total_reward(
                gap_reward, sim_reward, train_params, reduction=self.reduction
            )
        else:
            reward = -1 * gap_reward
        metrics = self.get_metrics(
            gap_reward, sim_reward, reward, uncertainty, final_state["smiles"]
        )
        self.environment.write_to_tensorboard(writer, num, **metrics)
        self.backprop(leaf_node, reward)

        self.save_outputs(final_state, metrics, num)
        return reward

    def run(self, load=False):
        """
        Executes the MCTS algorithm.

        Parameters:
        load (bool): If True, preloads the tree from a saved state.

        This method initializes the root state and root node of the tree. If the 
        `load` parameter is set to True, it preloads the tree from a saved state 
        and prints a confirmation message. It then runs a number of simulations 
        specified by `self.num_sims`, resetting the environment after each simulation.
        """
        root_state = self.environment.get_root_state()
        self.root = Tree_node(root_state, self.C, None, 0)

        if load:
            self.preload_tree()

            print("Done Loading")

        for i in range(self.num_sims):
            # try:
            current_reward = self.run_sim(i)
            # except:
            #     print("Failed! Skipping iteration")
            self.environment.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCTS for molecules")
    parser.add_argument("--output_dir", type=str, help="output folder")
    parser.add_argument("--environment", type=str, help="y6 or patent")
    parser.add_argument("--iter", type=int, help="iteration number")

    args = parser.parse_args()

    config = json.load(open(os.path.join(args.output_dir, "config.json")))
    train_params = config["train_params"]
    fname_params = config["fname_params"]

    iter_dir = os.path.join(args.output_dir, "iter_{}".format(args.iter))
    create_dir(iter_dir)
    TB_LOG_PATH = os.path.join(iter_dir, fname_params["tb_fname"])

    create_dir(TB_LOG_PATH)
    writer = SummaryWriter(TB_LOG_PATH)

    set_all_seeds(9999)
    environment = factory.create(
        args.environment,
        reward_tp=train_params["reward"],
        output_dir=args.output_dir,
        reduction=train_params["reduction"],
    )

    new_sim = MCTS(
        train_params["C"],
        environment=environment,
        exploration=train_params["exploration"],
        num_sims=train_params["num_sims"],
        reward_tp=train_params["reward"],
        reduction=train_params["reduction"],
    )
    new_sim.run()
    new_sim.save_tree(os.path.join(iter_dir, fname_params["root_node_fname"]))
