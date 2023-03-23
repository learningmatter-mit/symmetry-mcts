#!/usr/bin/env python
# coding: utf-8

import numpy as np
# import MCTS_io
import copy
import random
from molgen import react
import json
from chemprop.predict_one import predict_one
from torch.utils.tensorboard import SummaryWriter
from utils import create_dir
import argparse

def get_cores_side_chains_end_groups(json_path):
    f = json.load(open(json_path))
    cores = []
    side_chains_1 = []
    side_chains_2 = []
    end_groups = []

    for mol in f['molecules']:
        if mol['group'] == 'core':
            cores.append(mol)
        elif mol['group'] == 'side_chain_1':
            side_chains_1.append(mol)
        elif mol['group'] == 'side_chain_2':
            side_chains_2.append(mol)
        elif mol['group'] == 'end_group':
            end_groups.append(mol)
    return cores, end_groups, side_chains_1, side_chains_2

def get_next_actions_opd(state, cores, end_groups, side_chains_1, side_chains_2):
    new_actions = []

    if check_terminal(state):
        new_actions = []
    elif len(state.keys()) == 0:
        new_actions = cores
    elif ('He' in state['blocks'][0]['smiles']):
        new_actions = end_groups
    elif ('Ne' in state['blocks'][0]['smiles']):
        new_actions = side_chains_1
    elif ('Ar' in state['blocks'][0]['smiles']):
        new_actions = side_chains_2
    return new_actions

def check_compatibility(core, functional_group, pair_tuple):
    if (pair_tuple == ("a", "a")) and ("He" in core['blocks'][0]['smiles']) and ("He" in functional_group['blocks'][0]['smiles']):
        return True
    elif (pair_tuple == ("b", "b")) and ("Ne" in core['blocks'][0]['smiles']) and ("Ne" in functional_group['blocks'][0]['smiles']):
        return True
    elif (pair_tuple == ("c", "c")) and ("Ar" in core['blocks'][0]['smiles']) and ("Ar" in functional_group['blocks'][0]['smiles']):
        return True
    else:
        return False 

def check_terminal(state):
    if 'blocks' in state and len(state['blocks']) == 0:
        return 1
    return 0

class Tree_node():
    def __init__(self,state,C,parent,terminal):
        self.state = state
        self.is_terminal = terminal
        self.T = 0.0
        self.n = 0.0
        self.C = C
        self.parent = parent
        self.children = []
        self.complete_tree = False
        self.pre_val = 0
        self.pre_counter = 0
    
    def inc_T(self,val):
        self.T += val
    
    def inc_n(self):
        self.n += 1.0
        
    def get_n(self):
        return self.n
    
    def get_T(self):
        return self.T
    
    def get_v(self):
        if self.parent == None:
            return 1.0
        return (self.T/self.n)
    
    def inc_pre_val(self,reward):
        self.pre_val += reward
    
    def get_greedy(self):
        self.u = self.T/self.n
        return self.u

    def get_UCB(self, exploration, C):
        if exploration == 'random':
            return random.uniform(0, 1)
        else:
            if self.n == 0:
                if self.pre_counter > 0:
                    return self.pre_val/self.pre_counter + C*np.sqrt((2*np.log(self.parent.get_n()))/(1.0))
                else:
                    return C*np.sqrt((2*np.log(self.parent.get_n()))/(1.0))
            if self.parent == None:
                return 1.0
            self.u = (self.T/self.n) + C*np.sqrt((2*np.log(self.parent.get_n()))/(self.n))
            return self.u

class MCTS():

    def __init__(self, C, decay, cores, end_groups, side_chains_1, side_chains_2, property_target=0.77, property_bound=0.2, restricted = True, exploration='non_exp', num_sims=2500):
        
        self.C = C
        self.decay = decay
        self.cores = cores
        self.side_chains_1 = side_chains_1
        self.side_chains_2 = side_chains_2
        self.end_groups = end_groups
        self.root = Tree_node([],self.C,None,0)
        
        self.property_target = property_target
        self.property_bound = property_bound
        
        self.N_const = 1.0
        self.stable_structures_props = {}
        self.stable_structures_uncs = {}
        self.past_energies = {}
        self.candidates = {}
        self.count = []
        
        self.last_cat_cores = {}
        self.last_cat_side_chains_1 = {} # plus terminal condition
        self.last_cat_side_chains_2 = {}
        self.last_cat_end_groups = {}

        self.last_cat_cores_cache = {}
        self.last_cat_side_chains_1_cache = {}
        self.last_cat_side_chains_2_cache = {}
        self.last_cat_end_groups = {}

        self.prev_back = {}
        self.bias = {}
        self.num = 0
        self.exploration = exploration
        self.num_sims = num_sims

            
    def traverse(self,node,num, **kwargs):
        if (len(node.children) == 0) or (check_terminal(node.state)):
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
        next_actions = get_next_actions_opd(curr_state, self.cores, self.end_groups, self.side_chains_1, self.side_chains_2)
 
        next_nodes = []
        for na in next_actions:
            if na['group'] == 'core':
                next_state = na
                new_node = Tree_node(next_state, self.C, node, check_terminal(next_state))
                next_nodes.append(new_node)
                continue
            elif na['group'] == 'end_group':
                pair_tuple = ("a", "a")
            elif na['group'] == 'side_chain_1':
                pair_tuple = ("b", "b")
            elif na['group'] == 'side_chain_2':
                pair_tuple = ("c", "c")

            next_state = react.run('opd', core=curr_state, functional_group=na, reactive_pos=0, pair_tuple=pair_tuple) 
            next_nodes.append(next_state)
            new_node = Tree_node(next_state,self.C,node,check_terminal(next_state))
            node.children.append(new_node)
            continue
 
        move = np.random.randint(0,len(next_actions))
        return next_nodes[move]

    def roll_out(self,node, **kwargs):
        state = copy.deepcopy(node.state)
        while not check_terminal(state):
            next_actions = get_next_actions_opd(state,self.cores, self.end_groups, self.side_chains_1, self.side_chains_2)
            move = np.random.randint(0,len(next_actions))
            if (next_actions[move]['group'] == 'core'):
                next_action = next_actions[move]
                state = next_action
                continue
            elif next_actions[move]['group'] == 'end_group':
                pair_tuple = ("a", "a")
            elif next_actions[move]['group'] == 'side_chain_1':
                pair_tuple = ("b", "b")
            elif next_actions[move]['group'] == 'side_chain_2':
                pair_tuple = ("c", "c")
            state = react.run('opd', core=state, functional_group=next_actions[move], reactive_pos=0, pair_tuple=pair_tuple)[0]
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
        if leaf_node.get_n() != 0 and not check_terminal(leaf_node.state):
            leaf_node = self.expand(leaf_node)

        ### simulation/roll_out
        final_state = self.roll_out(leaf_node)
        prop, uncertainty = predict_one('models/all_db_checkpoints', [[final_state['smiles']]])
        prop = prop[0][1]
        uncertainty = uncertainty[0]

        reward = -1 * prop
        writer.add_scalar('prop_reward', reward, num)
        writer.add_scalar('uncertainty', uncertainty, num)
        if self.exploration == 'UCB_decay':
            writer.add_scalar('exploration coeff', self.C, num)
 
        self.backprop(leaf_node,reward)
        if final_state['smiles'] not in self.stable_structures_props.keys():
            self.stable_structures_props[final_state['smiles']] = prop
            self.stable_structures_uncs[final_state['smiles']] = uncertainty
 
        if num % 1 == 0:
            if self.exploration == 'UCB_decay' or self.exploration == 'random':
                with open('molecules_generated_prop_exploration_{}_num_sims_{}_decay_{}.json'.format(self.exploration, self.num_sims, self.decay), 'w') as f:
                    json.dump(self.stable_structures_props, f)
                with open('molecules_generated_uncs_exploration_{}_num_sims_{}_decay_{}.json'.format(self.exploration, self.num_sims, self.decay), 'w') as f:
                    json.dump(self.stable_structures_uncs, f)
            else:
                with open('molecules_generated_prop_exploration_{}_num_sims_{}_C_{}_decay_{}.json'.format(self.exploration, self.num_sims, self.C, self.decay), 'w') as f:
                    json.dump(self.stable_structures_props, f)
                with open('molecules_generated_uncs_exploration_{}_num_sims_{}_C_{}_decay_{}.json'.format(self.exploration, self.num_sims, self.C, self.decay), 'w') as f:
                    json.dump(self.stable_structures_uncs, f)
 
    def run(self, load=False):
        self.root = Tree_node({},self.C,None,0)
        self.candidates = {}
        self.count = []
        
        if load:
            self.preload_tree()
        
            print("Done Loading")
        
 
        for i in range(self.num_sims):
            self.run_sim(i)
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

    cores, end_groups, side_chains_1, side_chains_2 = get_cores_side_chains_end_groups('fragments/core-fxn-y6.json')
    C = args.C
    decay = args.decay
    exploration = args.exploration
    num_sims = args.num_sims

    TB_LOG_PATH = './runs_exploration_{}_num_sims_{}_C_{}_decay_{}/'.format(exploration, num_sims, C, decay)
    create_dir(TB_LOG_PATH)
    writer = SummaryWriter(TB_LOG_PATH)

    new_sim = MCTS(C, decay, cores, end_groups, side_chains_1, side_chains_2, exploration=exploration, num_sims=num_sims)
    new_sim.run()
