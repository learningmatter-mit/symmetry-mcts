#!/usr/bin/env python
# coding: utf-8

import numpy as np
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

    def __init__(self, C, decay, side_chains, end_groups, environment, property_target=0.77, property_bound=0.2, restricted = True, exploration='non_exp', num_sims=2500):
        
        self.C = C
        self.decay = decay
        self.side_chains = side_chains
        self.end_groups = end_groups
        self.root = Tree_node([],self.C,None,0)
        self.environment = environment
        
        self.property_target = property_target
        self.property_bound = property_bound
        
        self.N_const = 1.0
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

        reward = compute_molecular_mass(final_state['smiles'])
        print(reward)
        # wandb.log({
        #     "Iteration": num,
        #     "Reward": reward
        # })
        writer.add_scalar('molecular mass', reward, num)
        writer.add_scalar('num atoms', get_num_atoms(final_state['smiles']), num)
        writer.add_scalar('num carbon atoms', get_num_atoms_by_id(final_state['smiles'], 6), num)
        writer.add_scalar('num sulphur atoms', get_num_atoms_by_id(final_state['smiles'], 16), num)
        writer.add_scalar('num nitrogen atoms', get_num_atoms_by_id(final_state['smiles'], 7), num)

        # prop, uncertainty = predict_one('models/weights_lite', [[final_state['smiles']]])
        # prop = prop[0][0]
        # uncertainty = uncertainty[0]

        # reward = -1 * prop
        # writer.add_scalar('prop_reward', reward, num)
        # writer.add_scalar('uncertainty', uncertainty, num)
        # if self.exploration == 'UCB_decay':
        #     writer.add_scalar('exploration coeff', self.C, num)
 
        self.backprop(leaf_node,reward)
        # if final_state['smiles'] not in self.stable_structures_props.keys():
        #     self.stable_structures_props[final_state['smiles']] = prop
        #     self.stable_structures_uncs[final_state['smiles']] = uncertainty

        if final_state['smiles'] not in self.stable_structures_props.keys():
            self.stable_structures_props[final_state['smiles']] = reward

        if num % 1 == 0:
            if self.exploration == 'UCB_decay' or self.exploration == 'random':
                with open('molecules_generated_prop_exploration_{}_num_sims_{}_decay_{}.json'.format(self.exploration, self.num_sims, self.decay), 'w') as f:
                    json.dump(self.stable_structures_props, f)
                # with open('molecules_generated_uncs_exploration_{}_num_sims_{}_decay_{}.json'.format(self.exploration, self.num_sims, self.decay), 'w') as f:
                #     json.dump(self.stable_structures_uncs, f)
            else:
                with open('molecules_generated_prop_exploration_{}_num_sims_{}_C_{}_decay_{}.json'.format(self.exploration, self.num_sims, self.C, self.decay), 'w') as f:
                    json.dump(self.stable_structures_props, f)
                # with open('molecules_generated_uncs_exploration_{}_num_sims_{}_C_{}_decay_{}.json'.format(self.exploration, self.num_sims, self.C, self.decay), 'w') as f:
                #     json.dump(self.stable_structures_uncs, f)
 
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

    args = parser.parse_args()

    C = args.C
    decay = args.decay
    exploration = args.exploration
    num_sims = args.num_sims

    TB_LOG_PATH = './runs_exploration_{}_num_sims_{}_C_{}_decay_{}/'.format(exploration, num_sims, C, decay)
    create_dir(TB_LOG_PATH)
    writer = SummaryWriter(TB_LOG_PATH)

    set_all_seeds(9999)
    environment = Y6Environment()
    side_chains, end_groups = environment.get_side_chains_end_groups('fragments/core-fxn-y6-v2.json')

    # # start a new wandb run to track this script
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="MCTS",
    #     notes='Test run with Y6 derivatives',
    #     config={
    #     "C": C,
    #     "num_sims": num_sims
    #     } 
    # )

    new_sim = MCTS(C, decay, side_chains, end_groups, environment=environment, exploration=exploration, num_sims=num_sims)
    new_sim.run()
