#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import re
# import MCTS_io
import copy
import random
from molgen import react
import json
import argparse
import wandb

# from chemprop.predict_one import predict_one
from torch.utils.tensorboard import SummaryWriter
from utils import create_dir, compute_molecular_mass, get_num_atoms, get_num_atoms_by_id, set_all_seeds
from environment import Y6Environment
from tree_node import Tree_node
from rdkit import Chem
from rdkit.Chem import Descriptors


class MCTS:

    def __init__(self, C, decay, environment, property_target=0.77, property_bound=0.2, restricted=True, exploration='non_exp', num_sims=2500, reward_tp='bandgap', sweep_step=-1):
        
        self.C = C
        self.decay = decay
        self.root = Tree_node([],self.C,None,0)
        self.environment = environment
        
        self.property_target = property_target
        self.property_bound = property_bound
        
        self.N_const = 1.0

        self.stable_structures_dict = {}

        self.stable_structures_props = {}
        self.stable_structures_action_history = {}
        self.stable_structures_uncs = {}
        self.past_energies = {}
        self.candidates = {}
        self.count = []
        
        self.last_cat_side_chains = {} # plus terminal condition
        self.last_cat_end_groups = {}
        self.last_cat_pi_bridges = {}

        self.last_cat_side_chains_cache = {}
        self.last_cat_end_groups_cache = {}
        self.last_cat_pi_bridges_cache = {}

        self.prev_back = {}
        self.bias = {}
        self.num = 0
        self.exploration = exploration
        self.num_sims = num_sims
        self.reward_tp = reward_tp
        self.sweep_step = sweep_step
 
    def get_metrics(self, reward, uncertainty, smiles):
        if self.reward_tp == 'mass':
            metrics = {
                'molecular_mass': reward,
                'num_atoms': get_num_atoms(smiles),
                'num_carbon_atoms': get_num_atoms_by_id(smiles, 6),
                'num_sulphur_atoms': get_num_atoms_by_id(smiles, 16),
                'num_nitrogen_atoms': get_num_atoms_by_id(smiles, 7),
                'C': self.C
            }
        elif self.reward_tp == 'bandgap':
            metrics = {
                'reward': reward,
                'calibrated_gap': (1.592 * -1 * reward - 2.269),
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
            
            next_state_group = copy.deepcopy(curr_state['group'])
            next_state_group[action_group] += 1
            next_state['group'] = next_state_group

            next_state_fragments = copy.deepcopy(curr_state['fragments'])
            key = na.get_identifier()['key']
            identifier = na.get_identifier()['identifier']

            if key.startswith('pos') or key == 'end_group' or key == 'side_chain':
                next_state_fragments[key] = identifier
                next_state['fragments'] = next_state_fragments
            elif key.startswith('pi_bridge'):
                num_occurrences = len(next_state_fragments['pi_bridge_1']) + len(next_state_fragments['pi_bridge_2'])
                next_state_fragments[key + '_' + str(num_occurrences + 1)] = identifier
                next_state['fragments'] = next_state_fragments

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

            next_state_group = copy.deepcopy(state['group'])
            next_state_group[action_group] += 1
            next_state['group'] = next_state_group

            next_state_fragments = copy.deepcopy(state['fragments'])
            key = next_action.get_identifier()['key']
            identifier = next_action.get_identifier()['identifier']

            if key.startswith('pos') or key == 'end_group' or key == 'side_chain':
                next_state_fragments[key] = identifier
                next_state['fragments'] = next_state_fragments
            elif key.startswith('pi_bridge'):
                num_occurrences = int(next_state_fragments['pi_bridge_1'] != "") + int(next_state_fragments['pi_bridge_2'] != "")
                next_state_fragments[key + '_' + str(num_occurrences + 1)] = identifier
                next_state['fragments'] = next_state_fragments

            state = next_state
        return state
        
    def backprop(self,node,rw):
        node.inc_n()
        node.inc_T(rw)
        if node.parent != None:
            self.backprop(node.parent,rw)
            
    def run_sim(self,num):
        print("Iteration: ", num)
        self.num = num 
        ### selection
        curr_node = self.root
        
        leaf_node = self.traverse(curr_node, num)
        
        ### expansion and not check_terminal(leaf_node.state)
        if leaf_node.get_n() != 0 and not self.environment.check_terminal(leaf_node.state):
            leaf_node = self.expand(leaf_node)

        ### simulation/roll_out
        final_state = self.roll_out(leaf_node)

        try:
            reward, uncertainty = self.environment.get_reward(final_state['smiles'])
        except:
            import pdb; pdb.set_trace()
 
        metrics = self.get_metrics(reward, uncertainty, final_state['smiles'])
        self.environment.write_to_tensorboard(writer, num, **metrics)

        self.backprop(leaf_node,reward)

        if final_state['smiles'] not in self.stable_structures_props.keys():
            self.stable_structures_props[final_state['smiles']] = reward

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
            if self.exploration == 'UCB_decay' or self.exploration == 'random':
                with open('molecules_generated_prop_exploration_{}_num_sims_{}_decay_{}_reward_{}.json'.format(self.exploration, self.num_sims, self.decay, self.reward_tp), 'w') as f:
                    json.dump(self.stable_structures_props, f)
                df_stable_structures = pd.DataFrame.from_dict(self.stable_structures_dict)
                df_stable_structures.to_csv('molecules_generated_prop_exploration_{}_num_sims_{}_decay_{}_reward_{}.csv'.format(self.exploration, self.num_sims, self.decay, self.reward_tp), index=False)

                with open('molecules_generated_action_history_prop_exploration_{}_num_sims_{}_decay_{}_reward_{}.json'.format(self.exploration, self.num_sims, self.decay, self.reward_tp), 'w') as f:
                    json.dump(self.stable_structures_action_history, f)
            else:
                if self.sweep_step != -1:
                    fname = 'molecules_generated_prop_exploration_{}_num_sims_{}_C_{}_decay_{}_reward_{}_sweep_step_{}.json'.format(self.exploration, self.num_sims, self.C, self.decay, self.reward_tp, self.sweep_step)
                else:
                    fname = 'molecules_generated_prop_exploration_{}_num_sims_{}_C_{}_decay_{}_reward_{}.json'.format(self.exploration, self.num_sims, self.C, self.decay, self.reward_tp)
                
                with open(fname, 'w') as f:
                    json.dump(self.stable_structures_props, f)
                df = pd.DataFrame.from_dict(self.stable_structures_dict)
                df.to_csv(fname.replace('json', 'csv'), index=False)

    def run(self, load=False):
        root_state = self.environment.get_root_state() 
        self.root = Tree_node(root_state, self.C, None, 0)
        self.candidates = {}
        self.count = []
        
        if load:
            self.preload_tree()
        
            print("Done Loading")
 
        for i in range(self.num_sims):
            self.run_sim(i)
            self.environment.reset()
            if self.exploration == 'UCB_decay':
                self.C *= 0.99994
            self.count.append(len(self.stable_structures_props))
            if i % 10000 == 0:
                print("Iteration: " + str(i))

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='MCTS for molecules')
    parser.add_argument('--C', type=float, help='exploration coefficient', default=1.0)
    parser.add_argument('--decay', type=float, help='decay rate', default=1.0)
    parser.add_argument('--exploration', type=str, help='Type of exploration: UCB, random, UCB_decay, UCB_penalized')
    parser.add_argument('--num_sims', type=int, help='Number of simulations', default=2500)
    parser.add_argument('--reward', type=str, help='Type of reward: mass or bandgap')
    parser.add_argument('--sweep_step', type=int, help='sweep step if running parameter sweeps', default=-1)

    args = parser.parse_args()

    C = args.C
    decay = args.decay
    exploration = args.exploration
    num_sims = args.num_sims
    reward_tp = args.reward
    sweep_step = args.sweep_step

    if sweep_step != -1:
        TB_LOG_PATH = './runs_exploration_{}_num_sims_{}_C_{}_decay_{}_reward_{}_sweep_step_{}/'.format(exploration, num_sims, C, decay, reward_tp, sweep_step)
    else:
        TB_LOG_PATH = './runs_exploration_{}_num_sims_{}_C_{}_decay_{}_reward_{}/'.format(exploration, num_sims, C, decay, reward_tp)

    create_dir(TB_LOG_PATH)
    writer = SummaryWriter(TB_LOG_PATH)

    set_all_seeds(9999)
    environment = Y6Environment(reward_tp=args.reward)

    new_sim = MCTS(C, decay, environment=environment, exploration=exploration, num_sims=num_sims, reward_tp=reward_tp, sweep_step=sweep_step)
    new_sim.run()
"CCCCCCCCC(CCCCCC)Cc1c([conj])sc2c1oc1c2sc2c3sc4c(oc5c(CC(CCCCCC)CCCCCCCC)c([conj])sc54)c3c(F)c(F)c12"
"CCCCCCCCC(CCCCCC)Cc1c(C=c1sc([He])cc1)sc2c1oc1c2sc2c3sc4c(oc5c(CC(CCCCCC)CCCCCCCC)c([He])sc54)c3c(F)c(F)c12"