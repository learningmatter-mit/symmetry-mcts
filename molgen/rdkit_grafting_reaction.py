import collections
import itertools

from rdkit import Chem


def run_grafting_reactions(stock=None, scion=None, stock_site_groups=None,
                           scion_site_groups=None):
    # Get site groups for stock and scion.
    if not stock_site_groups:
        stock_site_groups = get_graft_site_groups_for_mol(stock)
    if not scion_site_groups:
        scion_site_groups = get_graft_site_groups_for_mol(scion)

    # Registry for products, to check for duplicates.
    product_smiles_registry = {}
    
    # Make pairs of compatible site groups
    # e.g. the '[*:1]' sites in the stock paired with.
    # the '[*:1]' sites in the scion.
    site_group_pairs = _get_graft_site_group_pairs(stock_site_groups,
                                                   scion_site_groups)
    for site_group, site_group_pair in site_group_pairs.items():
        stock_sites, scion_sites = site_group_pair
        # Make combinations of the scion sites, to match up with stock sites.
        # e.g. if the stock has sites A,B, and the scion has sites C,D,
        # we generate (A:C,B:C), (A:C,B:D), (A:D,B:C), (A:D,B:D). 
        scion_site_combos = itertools.product(scion_sites,
                                              repeat=len(stock_sites))
        for scion_site_combo in scion_site_combos:
            grafts = []
            for i, stock_site in enumerate(stock_sites):
                grafts.append({'scion': scion, 'stock_idx': stock_site,
                               'scion_idx': scion_site_combo[i]})
            product = make_grafts(stock=stock, grafts=grafts)
            product_smiles = Chem.MolToSmiles(product)
            if product_smiles not in product_smiles_registry:
                product_smiles_registry[product_smiles] = True
                yield product

def _get_graft_site_group_pairs(site_groups_1, site_groups_2):
    pairs = {}
    groups_in_both = set(site_groups_1.keys()).intersection(
        set(site_groups_2.keys()))
    for group in groups_in_both:
        pairs[group] = (site_groups_1[group], site_groups_2[group])
    return pairs

def get_graft_site_groups_for_mol(mol=None):
    sites = collections.defaultdict(list)
    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            map_number = atom.GetProp('molAtomMapNumber')
            sites[map_number].append(atom.GetIdx())
    return sites

def make_grafts(stock=None, grafts=None):
    working_stock = Chem.EditableMol(stock)
    idxs_to_remove = set()
    for graft in grafts:
        idxs_to_remove_for_graft = _add_graft(working_stock=working_stock,
                                              orig_stock=stock, graft=graft)
        idxs_to_remove.update(idxs_to_remove_for_graft)
    for idx_to_remove in reversed(sorted(idxs_to_remove)):
        working_stock.RemoveAtom(idx_to_remove)
    return working_stock.GetMol()

def _add_graft(working_stock=None, orig_stock=None, graft=None):
    # Copy molecules and bonds from scion to working stock.
    new_scion_idxs = []
    scion = graft['scion']
    for atom in scion.GetAtoms():
        new_idx = working_stock.AddAtom(atom)
        new_scion_idxs.append(new_idx)
    for bond in scion.GetBonds():
        working_stock.AddBond(new_scion_idxs[bond.GetBeginAtomIdx()],
                              new_scion_idxs[bond.GetEndAtomIdx()],
                              bond.GetBondType())

    # Create new bonds around graft sites.
    for stock_bond in orig_stock.GetAtomWithIdx(graft['stock_idx']).GetBonds():
        bond_begin_idx = stock_bond.GetOtherAtomIdx(graft['stock_idx'])
        for scion_bond in scion.GetAtomWithIdx(graft['scion_idx']).GetBonds():
            orig_scion_idx = scion_bond.GetOtherAtomIdx(graft['scion_idx'])
            bond_end_idx = new_scion_idxs[orig_scion_idx]
            working_stock.AddBond(bond_begin_idx, bond_end_idx,
                                  Chem.BondType.SINGLE)
    idxs_to_remove = set([graft['stock_idx'],
                          new_scion_idxs[graft['scion_idx']]])
    return idxs_to_remove
