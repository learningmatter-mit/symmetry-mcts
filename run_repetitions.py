import os
import time
import json
import subprocess
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
# from sbatchpy import run
# # from MCTS_y6 import fname_params

# config = {
#     "job-name": "train_mcts",
#     "partition": "xeon-p8",
#     "output": "train_mcts-%j.out",
#     "nodes": "1",
#     "ntasks-per-node": "20",
#     "cpus-per-task": "1",
#     "time": "96:00:00"
# }

SCRIPT_NAME='train_MCTS.sh'
LAST_N=100
OUTPUT_DIR='sum_reduction_5000'
START_ITER=0
NUM_REPETITIONS=50

# Function to generate Morgan fingerprints for a list of SMILES strings
def generate_morgan_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        fingerprint_array = np.array(fingerprint, dtype=int)
        return fingerprint_array
    else:
        return None


def is_job_running(job_id):
    try:
        # Run squeue and capture the output
        result = subprocess.run(['squeue', '--job', str(job_id)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # If the job is found in the squeue output, it is still running
        return 'JOBID' in result.stdout

    except subprocess.CalledProcessError as e:
        # Handle errors if necessary
        print(f"Error checking job status: {e}")
        return False


if __name__ == '__main__':
    config = json.load(open(os.path.join(OUTPUT_DIR, 'config.json')))
    fname_params = config['fname_params']
    if os.path.exists(os.path.join(OUTPUT_DIR.format(iter), 'best_smiles.csv')):
        df = pd.read_csv(os.path.join(OUTPUT_DIR.format(iter), 'best_smiles.csv'))
        best_molecules_list = list(df['best_smiles'])
        max_rewards_list = list(df['reward'])
        max_gap_rewards_list = list(df['gap_reward'])
        max_sim_rewards_list = list(df['sim_reward'])
    else:
        best_molecules_list = []
        max_rewards_list = []
        max_gap_rewards_list = []
        max_sim_rewards_list = []
    # slurm_job_pids = []
    for iter in range(START_ITER, NUM_REPETITIONS):
        slurm_process = subprocess.Popen(['sbatch', SCRIPT_NAME, OUTPUT_DIR, str(iter)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = slurm_process.communicate()
        slurm_job_pid = int(stdout.split()[-1])
        print(slurm_job_pid)
        # Wait for all SLURM jobs to finish
        # for slurm_pid in slurm_job_pids:
        while True:
            if not is_job_running(slurm_job_pid):
                break 
            # If the process still exists, wait for a short duration before checking again
            time.sleep(20)  # Adjust the sleep duration as needed
        df_results = pd.read_csv(os.path.join(OUTPUT_DIR, 'iter_{}'.format(iter), fname_params['molecules_fname']))
        # minimum_last_100 = min(df_results['reward'][-100:])

        # Get the last 100 rows
        last_100_rows = df_results.tail(LAST_N)
        reward = last_100_rows['reward']
        # reward = last_100_rows['gap_reward'] * last_100_rows['sim_reward']

        # Find the row where the 'gap' column has the minimum value among the last 100 rows
        max_reward_row = last_100_rows.loc[reward.idxmax()]

        # Access the value in the 'smiles' column for that row
        smiles_value = max_reward_row['smiles']
        max_reward = max_reward_row['reward']
        # max_reward = max_reward_row['gap_reward'] * max_reward_row['sim_reward']
        best_molecules_list.append(smiles_value)
        max_rewards_list.append(max_reward)
        max_gap_rewards_list.append(max_reward_row['gap_reward'])
        max_sim_rewards_list.append(max_reward_row['sim_reward'])
        df = pd.DataFrame({'best_smiles': best_molecules_list, 'reward': max_rewards_list, 'gap_reward': max_gap_rewards_list, 'sim_reward': max_sim_rewards_list})
        df.to_csv(os.path.join(OUTPUT_DIR.format(iter), 'best_smiles.csv'), index=False)

        fp = generate_morgan_fingerprint(smiles_value)
        if os.path.exists(os.path.join(OUTPUT_DIR, fname_params['fingerprints_fname'])):
            fingerprints = np.load(os.path.join(OUTPUT_DIR, fname_params['fingerprints_fname']))
        else:
            fingerprints = np.empty((0, 2048), dtype=int)
        fingerprints = np.vstack((fingerprints, fp))
        np.save(os.path.join(OUTPUT_DIR, fname_params['fingerprints_fname']), fingerprints)
