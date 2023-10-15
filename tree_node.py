import random
import numpy as np


class Tree_node():
    def __init__(self,state,C,parent,terminal):
        self.state = state
        self.is_terminal = terminal
        self.T = 0.0
        self.n = 0.0
        self.C = C
        self.fragment_counts = {'core': 0, 'pi_bridge': 0, 'end_group': 0, 'side_chain': 0}
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

    def inc_update_count(self, old_dict, key):
        self.fragment_counts = old_dict
        self.fragment_counts[key] += 1

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