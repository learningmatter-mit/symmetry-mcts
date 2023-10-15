import os, sys

filepath = os.path.realpath(__file__)
exepath = os.path.split(os.path.realpath(filepath))[0]
paths = [exepath, 'chemprop']
print(paths)
sys.path.insert(0, os.path.join(*paths))
sys.path.append(os.path.join(*paths))

import chemprop
from chemprop_inference import predict
# from chemprop.predict_one import predict_one
from utils import set_all_seeds

if __name__ == '__main__':
    set_all_seeds(9999)

    arguments = [
        '--test_path', 'chemprop_patent+mcts_train_smiles.csv',
        '--preds_path', 'chemprop_patent+mcts_train_smiles_preds.csv',
        '--uncertainty_method', 'ensemble',
        '--checkpoint_dir', 'models/patent_MCTS_checkpoints_ensemble'
    ]

    # arguments = [
    #     '--test_path', '/dev/null',
    #     '--preds_path', '/dev/null',
    #     '--uncertainty_method', 'ensemble',
    #     '--checkpoint_dir', 'models/patent_MCTS_checkpoints_ensemble',
    # ]

    # smiles = [['Cc1c(-c2ccc(-c3ccc(C=CC=CC=C4C(=O)c5cc(Br)c(Cl)cc5C4=C(C#N)C#N)c4nsnc34)s2)sc2c1oc1c2sc2c3sc4c(oc5c(C)c(-c6ccc(-c7ccc(C=CC=CC=C8C(=O)c9cc(Br)c(Cl)cc9C8=C(C#N)C#N)c8nsnc78)s6)sc54)c3c(F)c(F)c12']]
    args = chemprop.args.PredictArgs().parse_args(arguments)
    model_objects = chemprop.train.load_model(args=args)
    print(chemprop.train.make_predictions(args=args, model_objects=model_objects, return_uncertainty=True))
    # print(chemprop.train.make_predictions(args=args, smiles=smiles, model_objects=model_objects, return_uncertainty=True))
    # print(predict_one('models/patent_MCTS_checkpoints', [['Cc1c(-c2ccc(-c3ccc(C=CC=CC=C4C(=O)c5cc(Br)c(Cl)cc5C4=C(C#N)C#N)c4nsnc34)s2)sc2c1oc1c2sc2c3sc4c(oc5c(C)c(-c6ccc(-c7ccc(C=CC=CC=C8C(=O)c9cc(Br)c(Cl)cc9C8=C(C#N)C#N)c8nsnc78)s6)sc54)c3c(F)c(F)c12']]))
    # print(predict_one('models/patent_MCTS_checkpoints', [['Cc1c(-c2ccc(C=C3C(=O)c4c(csc4Cl)C3=C(C#N)C#N)c3nsnc23)n(C)c2c1oc1c2sc2c3sc4c(oc5c(C)c(-c6ccc(C=C7C(=O)c8c(csc8Cl)C7=C(C#N)C#N)c7nsnc67)n(C)c54)c3c3nccnc3c12']]))
