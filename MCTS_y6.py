#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import pandas as pd
import copy
import json
import argparse
import pstats

# from chemprop.predict_one import predict_one
from torch.utils.tensorboard import SummaryWriter
from utils import create_dir, get_num_atoms, get_num_atoms_by_id, set_all_seeds, get_total_reward, get_normalized_rewards
from environment import Y6Environment
from tree_node import Tree_node


class MCTS:

    def __init__(self, C, environment, exploration='UCB', num_sims=5000, reward_tp='bandgap', reduction='sum'):
        
        self.C = C
        self.root = Tree_node([],self.C,None,0)
        self.environment = environment
        
        self.stable_structures_dict = {}

        self.stable_structures_props = {}
        self.stable_structures_action_history = {}
        
        self.num_sims = num_sims
        self.reward_tp = reward_tp
        self.exploration = exploration
        self.reduction = reduction
 
    def process_next_state(self, curr_state, next_state, action_group, next_action):
        next_state_group = copy.deepcopy(curr_state['group'])
        next_state_group[action_group] += 1
        next_state['group'] = next_state_group

        next_state_fragments = copy.deepcopy(curr_state['fragments'])
        key = next_action.get_identifier()['key']
        identifier = next_action.get_identifier()['identifier']

        if key.startswith('pos') or key == 'end_group' or key == 'side_chain':
            next_state_fragments[key] = identifier
            next_state['fragments'] = next_state_fragments
        elif key.startswith('pi_bridge'):
            num_occurrences = int(len(next_state_fragments['pi_bridge_1']) != 0) + int(len(next_state_fragments['pi_bridge_2']) != 0)
            next_state_fragments[key + '_' + str(num_occurrences + 1)] = identifier
            next_state['fragments'] = next_state_fragments
        return next_state

    def save_outputs(self, final_state, metrics, num):
        if 'smiles' not in self.stable_structures_dict:
            self.stable_structures_dict['smiles'] = [final_state['smiles']]
        else:
            self.stable_structures_dict['smiles'].append(final_state['smiles'])

        if final_state['smiles'] not in self.stable_structures_action_history.keys():
            self.stable_structures_action_history[final_state['smiles']] = final_state['fragments']

        for key in metrics.keys():
            if key not in self.stable_structures_dict:
                self.stable_structures_dict[key] = [metrics[key]]
            else:
                self.stable_structures_dict[key].append(metrics[key])
 
        if num % 1 == 0:
            df_stable_structures = pd.DataFrame.from_dict(self.stable_structures_dict)
            df_stable_structures.to_csv(os.path.join(iter_dir, fname_params['molecules_fname']), index=False)

            with open(os.path.join(iter_dir, fname_params['action_history_fname']), 'w') as f:
                json.dump(self.stable_structures_action_history, f)

    def get_metrics(self, gap_reward, sim_reward, reward, uncertainty, smiles):
        if self.reward_tp == 'mass':
            metrics = {
                'molecular_mass': gap_reward,
                'num_atoms': get_num_atoms(smiles),
                'num_carbon_atoms': get_num_atoms_by_id(smiles, 6),
                'num_sulphur_atoms': get_num_atoms_by_id(smiles, 16),
                'num_nitrogen_atoms': get_num_atoms_by_id(smiles, 7),
                'C': self.C
            }
        elif self.reward_tp == 'bandgap' or self.reward_tp == 'tanimoto_bandgap':
            metrics = {
                'gap_reward': gap_reward,
                'sim_reward': sim_reward,
                'reward': reward,
                'C': self.C,
                'uncertainty': uncertainty
            }
        return metrics
 
    def traverse(self,node,num, **kwargs):
        if (len(node.children) == 0) or (self.environment.check_terminal(node.state)):
            return node
        else:
            if self.exploration == 'random':
                rand_index = np.random.randint(0, len(node.children))
                return self.traverse(node.children[rand_index], num)
            elif self.exploration == 'UCB':
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
    
    def expand(self,node, **kwargs):
        curr_state = node.state
        next_actions = self.environment.get_next_actions_opd(curr_state)
 
        next_nodes = []
        for na in next_actions:
            next_state, action_group = self.environment.propagate_state(curr_state, na)
            
            next_state = self.process_next_state(curr_state, next_state, action_group, na)
            new_node = Tree_node(next_state, self.C, node, self.environment.check_terminal(next_state))
            next_nodes.append(new_node)
            node.children.append(new_node)
 
        move = np.random.randint(0,len(next_actions))
        return next_nodes[move]

    def roll_out(self,node, **kwargs):
        state = copy.deepcopy(node.state)
        while not self.environment.check_terminal(state):
            next_actions = self.environment.get_next_actions_opd(state)
            move = np.random.randint(0,len(next_actions))
            next_action = next_actions[move]
            next_state, action_group = self.environment.propagate_state(state, next_action)

            next_state = self.process_next_state(state, next_state, action_group, next_action)
            state = next_state
        return state
        
    def backprop(self,node,rw):
        node.inc_n()
        node.inc_T(rw)
        if node.parent != None:
            self.backprop(node.parent,rw)
            
    def run_sim(self,num):
        print("Iteration: ", num)
        # selection
        curr_node = self.root
        leaf_node = self.traverse(curr_node, num)
        
        # expansion and not check_terminal(leaf_node.state)
        if leaf_node.get_n() != 0 and not self.environment.check_terminal(leaf_node.state):
            leaf_node = self.expand(leaf_node)

        # simulation/roll_out
        final_state = self.roll_out(leaf_node)
        gap_reward, sim_reward, uncertainty = self.environment.get_reward(final_state['smiles'])
        if os.path.exists(os.path.join(args.output_dir, 'fingerprints.npy')):
            reward = get_total_reward(gap_reward, sim_reward, train_params, reduction=self.reduction)
        else:
            # reward, _ = get_normalized_rewards(gap_reward, sim_reward, normalization_params)
            # reward = 1 - reward
            reward = -1 * gap_reward
        metrics = self.get_metrics(gap_reward, sim_reward, reward, uncertainty, final_state['smiles'])
        self.environment.write_to_tensorboard(writer, num, **metrics)
        self.backprop(leaf_node,reward)

        self.save_outputs(final_state, metrics, num)
        return reward

    def run(self, load=False):
        root_state = self.environment.get_root_state()
        self.root = Tree_node(root_state, self.C, None, 0)

        if load:
            self.preload_tree()

            print("Done Loading")

        for i in range(self.num_sims):
            current_reward = self.run_sim(i)
            self.environment.reset()
        # best_reward = float('-inf')  # Track the best reward achieved so far
        # plateau_counter = 0  # Counter for how many consecutive iterations the reward has not improved

        # for i in range(self.num_sims):
        #     current_reward = self.run_sim(i)
        #     self.environment.reset()

        #     # Check if the current reward is better than the best so far
        #     # rrent_reward = self.stable_structures_props.get(self.root.state['smiles'], float('-inf'))
        #     if current_reward > best_reward:
        #         best_reward = current_reward
        #         plateau_counter = 0
        #     else:
        #         plateau_counter += 1

        #     # If the reward has not improved for a certain number of iterations, exit the loop
        #     if plateau_counter >= train_params['patience_threshold']:
        #         print(f"Training terminated early as reward has plateaued for {train_params['patience_threshold']} iterations.")
        #         break

        #     if i % 10000 == 0:
        #         print("Iteration: " + str(i))

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='MCTS for molecules')
    parser.add_argument('--sweep_step', type=int, help='sweep step if running parameter sweeps', default=-1)
    parser.add_argument('--output_dir', type=str, help='output folder')
    parser.add_argument('--iter', type=int, help='iteration number')

    args = parser.parse_args()

    config = json.load(open(os.path.join(args.output_dir, 'config.json')))
    train_params = config['train_params']
    fname_params = config['fname_params']
    normalization_params = config['normalization_params']

    iter_dir = os.path.join(args.output_dir, 'iter_{}'.format(args.iter))
    create_dir(iter_dir)
    TB_LOG_PATH = os.path.join(iter_dir, fname_params['tb_fname'])

    create_dir(TB_LOG_PATH)
    writer = SummaryWriter(TB_LOG_PATH)

    set_all_seeds(9999)
    environment = Y6Environment(reward_tp=train_params['reward'], output_dir=args.output_dir, reduction=train_params['reduction'])

    new_sim = MCTS(train_params['C'], environment=environment, num_sims=train_params['num_sims'], reward_tp=train_params['reward'], reduction=train_params['reduction'])
    new_sim.run()
    # cProfile.run("new_sim.run()", filename='profile_stats_single_time_chemprop')
    
    # p = pstats.Stats("profile_stats_single_time_chemprop")
    # p.sort_stats("cumulative").print_stats()
    # # Print profiling statistics
    # with open('profile_stats', 'w') as f:
    #     pstats.Stats('profile_stats', stream=f).strip_dirs().sort_stats('cumulative').print_stats()
