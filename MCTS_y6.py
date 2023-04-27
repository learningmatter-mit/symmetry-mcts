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

from chemprop.predict_one import predict_one
from torch.utils.tensorboard import SummaryWriter
from utils import create_dir, compute_molecular_mass, get_num_atoms, get_num_atoms_by_id, set_all_seeds
from environment import Y6Environment
from tree_node import Tree_node
from rdkit import Chem
from rdkit.Chem import Descriptors


class MCTS:

    def __init__(self, C, decay, side_chains, end_groups, environment, property_target=0.77, property_bound=0.2, restricted=True, exploration='non_exp', num_sims=2500, reward_tp='bandgap'):
        
        self.C = C
        self.decay = decay
        self.side_chains = side_chains
        self.end_groups = end_groups
        self.root = Tree_node([],self.C,None,0)
        self.environment = environment
        
        self.property_target = property_target
        self.property_bound = property_bound
        
        self.N_const = 1.0

        self.stable_structures_dict = {}

        self.stable_structures_props = {}
        self.stable_structures_uncs = {}
        self.past_energies = {}
        self.candidates = {}
        self.count = []
        
        self.last_cat_side_chains = {} # plus terminal condition
        self.last_cat_end_groups = {}

        self.last_cat_side_chains_cache = {}
        self.last_cat_end_groups_cache = {}

        self.prev_back = {}
        self.bias = {}
        self.num = 0
        self.exploration = exploration
        self.num_sims = num_sims
        self.reward_tp = reward_tp
 
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
        next_actions = self.environment.get_next_actions_opd(curr_state, self.side_chains, self.end_groups)
 
        next_nodes = []
        for na in next_actions:
            next_state = self.environment.propagate_state(curr_state, na)
            new_node = Tree_node(next_state, self.C, node, self.environment.check_terminal(next_state))
            next_nodes.append(new_node)
            node.children.append(new_node)
 
        move = np.random.randint(0,len(next_actions))
        return next_nodes[move]

    def roll_out(self,node, **kwargs):
        state = copy.deepcopy(node.state)
        while not self.environment.check_terminal(state):
            next_actions = self.environment.get_next_actions_opd(state, self.side_chains, self.end_groups)
            move = np.random.randint(0,len(next_actions))
            next_action = next_actions[move]
            next_state = self.environment.propagate_state(state, next_action)
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

        reward, uncertainty = self.environment.get_reward(final_state['smiles'])

 
        metrics = self.get_metrics(reward, uncertainty, final_state['smiles'])
        self.environment.write_to_tensorboard(writer, num, **metrics)

        self.backprop(leaf_node,reward)

        if final_state['smiles'] not in self.stable_structures_props.keys():
            self.stable_structures_props[final_state['smiles']] = reward

        if 'smiles' not in self.stable_structures_dict:
            self.stable_structures_dict['smiles'] = [final_state['smiles']]
        else:
            self.stable_structures_dict['smiles'].append(final_state['smiles'])


        for key in metrics.keys():
            if key not in self.stable_structures_dict:
                self.stable_structures_dict[key] = [metrics[key]]
            else:
                self.stable_structures_dict[key].append(metrics[key])
 
        if num % 1 == 0:
            if self.exploration == 'UCB_decay' or self.exploration == 'random':
                with open('molecules_generated_prop_exploration_{}_num_sims_{}_decay_{}_reward_{}.json'.format(self.exploration, self.num_sims, self.decay, self.reward_tp), 'w') as f:
                    json.dump(self.stable_structures_props, f)
                df = pd.DataFrame.from_dict(self.stable_structures_dict)
                df.to_csv('molecules_generated_prop_exploration_{}_num_sims_{}_decay_{}_reward_{}.csv'.format(self.exploration, self.num_sims, self.decay, self.reward_tp), index=False)
            else:
                with open('molecules_generated_prop_exploration_{}_num_sims_{}_C_{}_decay_{}_reward_{}.json'.format(self.exploration, self.num_sims, self.C, self.decay, self.reward_tp), 'w') as f:
                    json.dump(self.stable_structures_props, f)
                df = pd.DataFrame.from_dict(self.stable_structures_dict)
                df.to_csv('molecules_generated_prop_exploration_{}_num_sims_{}_C_{}_decay_{}_reward_{}.csv'.format(self.exploration, self.num_sims, self.C, self.decay, self.reward_tp), index=False)

    def run(self, load=False):
        root_state = {
            'smiles': "c1cc2<pos2>c3c4c5n<pos0>nc5c6c7<pos2>c8cc<pos3>c8c7<pos1>c6c4<pos1>c3c2<pos3>1",
            'label': 'opd',
            'group': 'zero',
            'blocks': [{'smiles': 'c1([He])c([Ne])c2<pos2>c3c4c5n<pos0>nc5c6c7<pos2>c8c([Ne])c([He])<pos3>c8c7<pos1>c6c4<pos1>c3c2<pos3>1'}]
        }
        self.root = Tree_node(root_state, self.C, None, 0)
        self.candidates = {}
        self.count = []
        
        if load:
            self.preload_tree()
        
            print("Done Loading")
        
 
        for i in range(self.num_sims):
            self.run_sim(i)
            self.C *= 0.9994
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

    args = parser.parse_args()

    C = args.C
    decay = args.decay
    exploration = args.exploration
    num_sims = args.num_sims
    reward_tp = args.reward

    TB_LOG_PATH = './runs_exploration_{}_num_sims_{}_C_{}_decay_{}_reward_{}/'.format(exploration, num_sims, C, decay, reward_tp)
    create_dir(TB_LOG_PATH)
    writer = SummaryWriter(TB_LOG_PATH)

    set_all_seeds(9999)
    environment = Y6Environment(reward_tp=args.reward)
    side_chains, end_groups = environment.get_side_chains_end_groups('fragments/core-fxn-y6-v2.json')

    new_sim = MCTS(C, decay, side_chains, end_groups, environment=environment, exploration=exploration, num_sims=num_sims, reward_tp=reward_tp)
    new_sim.run()
