import copy
from molgen.mol_storage import MolStorage
from molgen.molecules import OPDMolecules

def opd_builder(core, functional_group, reactive_pos, pair_tuple):
    core_new = copy.deepcopy(core)
    core_new['blocks'] = [core_new['blocks'][reactive_pos]]
    core_storage = MolStorage(core_new["smiles"], core_new["label"],
                                core_new["group"], core_new["blocks"])
    functional_group_storage = MolStorage(functional_group["smiles"], functional_group["label"],
                                            functional_group["group"], functional_group["blocks"])
    return OPDMolecules(core_storage, functional_group_storage, pair_tuple)
