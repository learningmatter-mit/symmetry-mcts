import random
import numpy as np


class Tree_node:
    """
    A class used to represent a node in a Monte Carlo Tree Search (MCTS).

    Attributes
    ----------
    state : any
        The state associated with this node.
    is_terminal : bool
        Indicates if the node is terminal.
    T : float
        The total value of the node.
    n : float
        The visit count of the node.
    C : float
        The exploration constant.
    fragment_counts : dict
        A dictionary to count different types of fragments.
    fragment_identities : dict
        A dictionary to store identities of different fragments.
    parent : Tree_node or None
        The parent node.
    children : list
        A list of child nodes.
    complete_tree : bool
        Indicates if the tree is complete.
    pre_val : float
        The pre-calculated value.
    pre_counter : int
        The counter for pre-calculated values.

    Methods
    -------
    inc_T(val)
        Increments the total value T by a given value.
    inc_n()
        Increments the visit count n by 1.
    get_n()
        Returns the visit count n.
    get_T()
        Returns the total value T.
    inc_update_count(old_dict, key)
        Updates the fragment count dictionary and increments the count for a given key.
    get_v()
        Returns the value of the node. If the node has no parent, returns 1.0.
    inc_pre_val(reward)
        Increments the pre-calculated value by a given reward.
    get_greedy()
        Returns the greedy value of the node.
    get_UCB(exploration, C)
        Returns the Upper Confidence Bound (UCB) value of the node based on the exploration strategy.
    """
    def __init__(self, state, C, parent, terminal):
        self.state = state
        self.is_terminal = terminal
        self.T = 0.0
        self.n = 0.0
        self.C = C
        self.fragment_counts = {
            "core": 0,
            "pi_bridge": 0,
            "end_group": 0,
            "side_chain": 0,
        }
        self.fragment_identities = {
            "pos0": {},
            "pos1": {},
            "pos2": {},
            "pos3": {},
            "pi_bridge1": {},
            "pi_bridge2": {},
            "end_group": {},
            "side_chain": {},
        }
        self.parent = parent
        self.children = []
        self.complete_tree = False
        self.pre_val = 0
        self.pre_counter = 0

    def inc_T(self, val):
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
        return self.T / self.n

    def inc_pre_val(self, reward):
        self.pre_val += reward

    def get_greedy(self):
        self.u = self.T / self.n
        return self.u

    def get_UCB(self, exploration, C):
        """
        Calculate the Upper Confidence Bound (UCB) for the current node.

        Parameters:
        exploration (str): The exploration strategy to use. If "random", a random value between 0 and 1 is returned.
        C (float): The exploration parameter that balances exploration and exploitation.

        Returns:
        float: The UCB value for the current node.

        Notes:
        - If the exploration strategy is "random", a random value between 0 and 1 is returned.
        - If the node has not been visited (self.n == 0) and has a pre-counter value greater than 0, the UCB is calculated using the pre_val and pre_counter.
        - If the node has not been visited (self.n == 0) and has no pre-counter value, the UCB is calculated using only the exploration parameter C.
        - If the node has no parent, a default value of 1.0 is returned.
        - Otherwise, the UCB is calculated using the node's total value (self.T), visit count (self.n), and the parent's visit count.
        """
        if exploration == "random":
            return random.uniform(0, 1)
        else:
            if self.n == 0:
                if self.pre_counter > 0:
                    return self.pre_val / self.pre_counter + C * np.sqrt(
                        (2 * np.log(self.parent.get_n())) / (1.0)
                    )
                else:
                    return C * np.sqrt((2 * np.log(self.parent.get_n())) / (1.0))
            if self.parent == None:
                return 1.0
            self.u = (self.T / self.n) + C * np.sqrt(
                (2 * np.log(self.parent.get_n())) / (self.n)
            )
            return self.u
