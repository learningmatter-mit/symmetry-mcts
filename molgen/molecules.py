
import rdkit
import numpy as np

from rdkit import Chem
from rdkit.Chem import rdmolfiles

from molgen.blocks.blockset import BlockSet
from molgen.blocks import Block

from molgen.blockreactor import rxn_envs, cleaner_env, old_style_rxn_envs, old_cleaner_env
from molgen.blockreactor import make_cleaner_func
from molgen.blockreactor import mass_under, contains_not_smarts


def add_reactive_sites(smi, num_sites, multiple=False):
    smis_ = set()
    # initiate the atom to add, which is a He
    at = Chem.AtomFromSmiles('C')
    at.SetAtomicNum(2)
    at2 = Chem.AtomFromSmiles('C')
    at2.SetAtomicNum(2)
    ats = [at, at2]
    # 
    try:
#     pdb.set_trace()
        m = Chem.MolFromSmiles(smi)
    #     m.calcImplicitValence()
        # m = Chem.MolFromSmiles('C1=CC=CC=C1')
        z = list(rdmolfiles.CanonicalRankAtoms(m, breakTies=False))
        matches = m.GetSubstructMatches(m, uniquify=False)
        if len(matches) >= num_sites:
            a = np.array(matches)
            pairs = np.unique(np.sort(a.T), axis=0).tolist()
            for pair in pairs:
                pair = list(set(pair))
                signal = True
                if len(pair) > num_sites:
                    if num_sites == 2:
                        pathl_dict = {}
    #                         print(pair)
                        for i in range(1, len(pair)):
                            pathl = Chem.GetShortestPath(m, pair[0], pair[i])
    #                             print(pathl)
                            pathl_dict.update({pair[i]:len(pathl)})
                        items_ = sorted(pathl_dict.items(), key=operator.itemgetter(1))
                        pair = np.array([pair[0], items_[-1][0]])
                        pair.astype(int)
                        pair = pair.tolist()
                    elif num_sites == 1:
                        pair = pair[:1]
                    elif num_sites == 0:
                        break
                elif len(pair) == num_sites and num_sites > 1:
                    if pair[0] == pair[1]:
                        signal = False
    #                         print('pair is now ' + str(pair))
                if (m.GetAtomWithIdx(pair[0]).GetAtomicNum() == 6 or m.GetNumAtoms() == 1) and \
                m.GetAtomWithIdx(pair[0]).GetNumImplicitHs() > 0 and \
                (m.GetAtomWithIdx(pair[0]).GetIsAromatic() or num_sites == 1) and \
                signal:
    #                     print(pair)
                    m_ = Chem.EditableMol(m)
                    for i, atidx in enumerate(pair):
                        atidx_ = m_.AddAtom(ats[i])
                        m_.AddBond(atidx, atidx_, rdkit.Chem.rdchem.BondType.SINGLE)
                    m_ = m_.GetMol()
                    smi_ = Chem.MolToSmiles(m_)
                    smis_.add(smi_)
            return list(smis_)
        else:
            print('%s does not contain at least %s symmetric reactive sites.' %(smi, num_sites))
            return []
    except:
        print('Mol initiation failed for %s' %(smi))
        return []


class OPDMolecules:
    def __init__(self, core_storage, functional_group_storage, pair_tuple):
        self.core = core_storage
        self.functional_group = functional_group_storage
        self.pair_tuple = pair_tuple
        self.cleaners = {
            "1": [
                "[*:1][He]>>[*:1]",
                "[*:1][Ne]>>[*:1]",
                "[*:1][Ar]>>[*:1]"
            ]
        }

    def update_with_reactive_sites(self, smiles_list):
        updated_list = []
        if self.pair_tuple == ('a', 'a'):
            group = 'stage_1'
        elif self.pair_tuple == ('b', 'b'):
            group = 'stage_2'
        elif self.pair_tuple == ('c', 'c'):
            group = 'stage_3'

        for smiles in smiles_list:
            if group == 'stage_3':
                blocks = []
            else:
                blocks = [{'smiles': smiles}]
            
            molecule = {
                'smiles': rxn_envs[self.pair_tuple].clean_smiles(smiles),
                'label': 'opd',
                'group': group,
                'blocks' : blocks
            }
            updated_list.append(molecule)

        return updated_list

    def react(self, clean=False):
        cleaner_funcs = []
        for marker_number, smarts_list in self.cleaners.items():
            cleaner_funcs.append(make_cleaner_func(smarts_list, marker_number))

        cores = BlockSet(self.core.get_blocks())
        functional_groups = BlockSet(self.functional_group.get_blocks())
 
        single_pass = BlockSet()
        with rxn_envs[self.pair_tuple]:
            single_pass.update(cores.react_sym(functional_groups))

        single_pass_smiles = [block.smiles() for block in single_pass]
        smiles_dicts = self.update_with_reactive_sites(single_pass_smiles)
 
        return smiles_dicts
