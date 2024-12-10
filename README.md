# Symmetry-Constrained Monte Carlo Tree Search
[![arXiv](https://img.shields.io/badge/arXiv-2410.08833-84cc16)](https://arxiv.org/abs/2410.08833)
[![MIT](https://img.shields.io/badge/License-MIT-3b82f6.svg)](https://opensource.org/license/mit)

This repository contains code to train a symmetry-constrained monte-carlo tree search (MCTS) algorithm which uses fragments obtained from patent-mined data. Details about the method and results can be found at [Symmetry-Constrained Generation of Diverse Low-Bandgap Molecules with Monte Carlo Tree Search](https://arxiv.org/abs/2410.08833). Details about how the patent data was obtained can be found at [Automated patent extraction powers generative modeling in focused chemical spaces](https://pubs.rsc.org/en/content/articlehtml/2023/dd/d3dd00041a).

## Setup and Installation
All packages are provided in `environment.yml` and can be installed with: 

`conda env create --name mcts --file=environment.yml` 

Place a config file `config.json` inside a folder (say `test_folder`) containing training parameters. We provide a sample version in `test_folder`.

### Chemprop Installation
```
git clone git@github.com:chemprop/chemprop.git
cd chemprop && pip install -e .
```

### Symmetry-Informed Fragment Decomposition
Code to preprocess, cluster and decompose fragments from patents can be found in `fragments/`. We provide the output files from these steps so that training can be done without having to run these first.

## Training
### Chemprop
Example code to train chemprop can be found in `train_chemprop/`. We already provide checkpoints, so re-training is not necessary to reproduce MCTS training.

### MCTS
The following command will run one training of the MCTS model. 

`python train_mcts.py --output_dir test_folder --environment patent --iter 0`

If you would like to run multiple iterations of training to generate diverse candidates (see more details in the paper), you can run training repetitions. A simple example of how this can be done in a SLURM-based cluster can be found in `exammple_scripts/run_repetitions.py`.

## Citation
If you use code from this repository, please cite:
```
@article{subramanian2024symmetry,
  title={Symmetry-Constrained Generation of Diverse Low-Bandgap Molecules with Monte Carlo Tree Search},
  author={Subramanian, Akshay and Damewood, James and Nam, Juno and Greenman, Kevin P and Singhal, Avni P and G{\'o}mez-Bombarelli, Rafael},
  journal={arXiv preprint arXiv:2410.08833},
  year={2024}
}
```
If you use the patent-mined datasets, please additionally cite:
```
@article{subramanian2023automated,
  title={Automated patent extraction powers generative modeling in focused chemical spaces},
  author={Subramanian, Akshay and Greenman, Kevin P and Gervaix, Alexis and Yang, Tzuhsiung and G{\'o}mez-Bombarelli, Rafael},
  journal={Digital Discovery},
  volume={2},
  number={4},
  pages={1006--1015},
  year={2023},
  publisher={Royal Society of Chemistry}
}
```
