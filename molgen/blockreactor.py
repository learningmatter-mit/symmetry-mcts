import itertools
from rdkit.Chem.AllChem import MolFromSmarts, CalcExactMolWt

from molgen.blocks.reactionenvironment import RDKitReactionEnvironment
from molgen.blocks.protect import ExposeOnly
from molgen.blocks.blockset import BlockSet
from .singlereactant_rxnenv import SingleReactantRxnEnv

ALPHA_ATOM = "He"
BETA_ATOM = "Ne"
GAMMA_ATOM = "Ar"
DELTA_ATOM = "Kr"
EPS_ATOM = "Xe"
ZETA_ATOM = "Rn"

MARKER_KEY = "molAtomMapNumber"


class MyReactionEnvironment(RDKitReactionEnvironment):
    def estimate(self, left_block_site_count, right_block_site_count):
        # add to right side set to account for empty option on left
        return (right_block_site_count + 1) ** left_block_site_count - 1

    def estimate_sym(self, left_block_site_count, right_block_site_count):
        # add to right side set to account for empty option on left
        return right_block_site_count


def smarts(left_gender_atom, right_gender_atom, mode='linking'):
    if mode == 'linking':
        base = "[*:1][{lga}:3].[*:2][{rga}:4]" + \
               ">>[*:1][*:2].[{lga}:3][{rga}:4]"
    elif mode == 'fusion':
        base = "[*:4]~[c:1]:[c:2]([{lga}:3])~[*:5].[a;H1:7]~[a:8]-[{rga}:9]" + \
               ">>[*:3][*:9].[*:4]~[*:7]:[*:8]~[*:5]"
    elif mode == 'double':
        base = "[*;X4:1]([{lga}:3])[{lga}:4].[*;X4:2]([{rga}:5])[{rga}:6]" + \
               ">>[*:1]=[*:2].[*:3][*:4][*:5][*:6]"
    elif mode == 'spiro':
        base = "[*:1]([{lga}:3])[{lga}:4].[*:2]([{rga}:5])([{rga}:6])([*:7])[*:8]" + \
               ">>[*:1]([*:7])[*:8].[*:2][*:3][*:4][*:5][*:6]"
    else:
        raise ValueError(
            'Mode should be either linking, fusion, double or spiro')
    return base.format(lga=left_gender_atom, rga=right_gender_atom)


def get_rxn_env_dict(a_atom, b_atom, c_atom, d_atom, e_atom, f_atom, mode='linking'):
    """
    return a dictionary of reaction environments that reflect all possible combinations of reactive sites
    the dictionary allows lookup of the environments with by tuple, such as ("a", "b") or ("c", "c")
    """
    env_dict = {}
    handle_dict = {"a": a_atom, "b": b_atom, "c": c_atom,
                   "d": d_atom, "e": e_atom, "f": f_atom}
    for name in itertools.product("abcdef", repeat=2):
        smarts_str = smarts(
            handle_dict[name[0]], handle_dict[name[1]], mode=mode)
        env_dict[name] = MyReactionEnvironment(smarts_str, handle_list)

    return env_dict


handle_list = [ALPHA_ATOM, BETA_ATOM,
               GAMMA_ATOM, DELTA_ATOM, EPS_ATOM, ZETA_ATOM]
rxn_envs = get_rxn_env_dict(*handle_list)
cleaner_env = MyReactionEnvironment("", handle_list)


################
##   OLD ATOMS  ##
################


OLD_ALPHA_ATOM = "At"
OLD_BETA_ATOM = "Be"
OLD_GAMMA_ATOM = "Cd"
OLD_DELTA_ATOM = "Dy"

MARKER_KEY = "molAtomMapNumber"


old_handle_dict = {"a": OLD_ALPHA_ATOM, "b": OLD_BETA_ATOM,
                   "c": OLD_GAMMA_ATOM, "d": OLD_DELTA_ATOM}
old_handle_list = list(old_handle_dict.values())
# make all rxn_envs with full handle list as it's just used for cleaning, and we want to clean everything regardless
old_cleaner_env = MyReactionEnvironment("", old_handle_list)

old_style_rxn_envs = {}
for name in itertools.product("abcd", repeat=2):
    smarts_str = smarts(old_handle_dict[name[0]], old_handle_dict[name[1]])
    old_style_rxn_envs[name] = MyReactionEnvironment(
        smarts_str, old_handle_list)


################
##   FILTERS  ##
################

def mass_under(mass):
    def filter(block):
        return CalcExactMolWt(block.cleaned().mol) < mass
    return filter


def contains_not_smarts(smarts_list):
    submols = [MolFromSmarts(smarts) for smarts in smarts_list]

    def filter(block):
        for submol in submols:
            if block.mol.HasSubstructMatch(submol):
                return False
        return True
    return filter

################
##   Tools    ##
################


def conditional_react_fill(prod, marker):
    marked_sites = []
    for site in prod.sites():
        has_mark = False
        for atom in site.atoms():
            if atom.HasProp(MARKER_KEY) and atom.GetProp(MARKER_KEY) == marker:
                has_mark = True
        if has_mark:
            marked_sites.append(site)
    if len(marked_sites) > 0:
        with ExposeOnly(*marked_sites):
            return BlockSet(prod.react_fill())
    else:
        return BlockSet()


def make_cleaner_func(smarts_list, marker_value):
    """
    create a function that cleans sites using the given list of smarts strings,
    but only if they are marked with the integer marker_value
    """
    rxn_env_list = [SingleReactantRxnEnv(
        str(smarts), handle_list) for smarts in smarts_list]
    marker = str(marker_value)

    def cleaner_func(prod):
        """ return cleaned version of product if possible, otherwise just return prod """
        out_set = BlockSet()
        in_blocks = [prod]
        for rxn_env in rxn_env_list:
            with rxn_env:
                for b in in_blocks:
                    try:
                        results = conditional_react_fill(b, marker)
                        if len(results) == 0:
                            out_set.add(b)
                        else:
                            out_set |= results
                    except (RuntimeError, ValueError):
                        print(b.smiles(), rxn_env, "blew it")
                        raise
            in_blocks = out_set
            out_set = BlockSet()
        clean_prod_set = in_blocks
        if len(clean_prod_set) > 1:
            raise Exception(
                "Ambiguous cleaning process: Cleaner produces more than one possible result")
        cleaned_prod = clean_prod_set.pop()
        cleaned_prod.take_heritage(prod)  # override original heritage
        return cleaned_prod
    return cleaner_func
