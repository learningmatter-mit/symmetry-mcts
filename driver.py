import argparse
import subprocess
import itertools

parser = argparse.ArgumentParser(description='driver code for MCTS')
parser.add_argument('--sweep_step', type=int, help='sweep step', required=True)

args = parser.parse_args()

# C_list = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
C_list = [0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 1]
# decay = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
# combs = list(itertools.product(C, decay))
sweep_step = int(args.sweep_step)
C = C_list[sweep_step]
# C, decay = combs[sweep_step]
# print(C, decay)

subprocess.check_call('python MCTS_y6.py'
                        ' --C {}'
                        ' --exploration {}'
                        ' --num_sims {}'
                        ' --reward {}'
                        ' --sweep_step {}'.format(C, 'UCB', 5000, 'bandgap', sweep_step), shell=True)