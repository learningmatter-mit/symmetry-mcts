import weakref

from .reactivesite import ReactiveSite

__protect_stack__ = []
PROTECT_BASE_NUM = 100
# this is a special property for rdkit that will be maintained through a reaction
PROTECT_PROP_KEY = "molAtomMapNumber"
PROTECTED_KEY = '_protected'


def protect_stack_push(cm):
    global __protect_stack__
    __protect_stack__.append(cm)


def protect_stack_pop(cm):
    global __protect_stack__
    return __protect_stack__.pop()


def protect_stack_height():
    global __protect_stack__
    return len(__protect_stack__)


def current_protect():
    global __protect_stack__
    if len(__protect_stack__) > 0:
        return __protect_stack__[-1]
    else:
        return None


def base_protect():
    global __protect_stack__
    if len(__protect_stack__) > 0:
        return __protect_stack__[0]
    else:
        return None


def protect_stack():
    global __protect_stack__
    return __protect_stack__


def honor_protected_atoms(react_func):
    def protected(self, *blocks, **kwargs):
        # we will check for reactability against the base
        base_protector = base_protect()
        complete_protect_stack = protect_stack()
        if base_protector is not None:
            base_protector.protect_blocks(blocks)
            try:
                products = react_func(self, *blocks, **kwargs)
                for protector in complete_protect_stack:
                    protector.register_blocks(products)
                base_protector.unprotect_blocks(products)
            finally:
                base_protector.unprotect_blocks(blocks)
        else:
            products = react_func(self, *blocks, **kwargs)
        return products
    return protected


class Protect:
    """
    Context Manager to only allow reaction with the given site for a
    given block site is a tuple of atom indices for this block
    """
    def __init__(self, *sites):
        self._sites = sites
        self._protect_val = 0
        self._cleanup_blocks = []

    def __enter__(self):
        self._protect_val = 10**(protect_stack_height()) * PROTECT_BASE_NUM
        protect_stack_push(self)
        for site in self._sites:
            for atom in site.atoms():
                self._mark_atom(atom)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        for site in self._sites:
            for atom in site.atoms():
                self._unmark_atom(atom)
        for bloc_wref in self._cleanup_blocks:
            for atom in bloc_wref().atoms():
                self._unmark_atom(atom)
        self._cleanup_blocks = []
        protect_stack_pop(self)
                #site.block.mol.GetAtomWithIdx(atom_idx).ClearProp('_protected')

    def _mark_atom(self, atom):
        if atom.HasProp(PROTECT_PROP_KEY):
            curval = int(atom.GetProp(PROTECT_PROP_KEY))
            atom.SetProp(PROTECT_PROP_KEY, str(curval + self._protect_val))
        else:
            atom.SetProp(PROTECT_PROP_KEY, str(self._protect_val))

    def _unmark_atom(self, atom):
        if atom.HasProp(PROTECT_PROP_KEY):
            curval = int(atom.GetProp(PROTECT_PROP_KEY))
            if curval >= self._protect_val:
                remainder = curval - self._protect_val
                if remainder > 0:
                    atom.SetProp(PROTECT_PROP_KEY, str(remainder))
                else:
                    atom.ClearProp(PROTECT_PROP_KEY)

    def is_protected(self, atom):
        if atom.HasProp(PROTECT_PROP_KEY):
            val = int(atom.GetProp(PROTECT_PROP_KEY))
            if val >= self._protect_val:
                return True
        else:
            return False

    def protect_blocks(self, blocks):
        for bloc in blocks:
            for atom in bloc.atoms():
                if atom.HasProp(PROTECT_PROP_KEY):
                    val = int(atom.GetProp(PROTECT_PROP_KEY))
                    if val >= self._protect_val:
                        atom.SetProp(PROTECTED_KEY, '1')

    def unprotect_blocks(self, blocks):
        for bloc in blocks:
            for atom in bloc.atoms():
                if atom.HasProp(PROTECTED_KEY):
                    atom.ClearProp(PROTECTED_KEY)

    def register_blocks(self, blocks):
        for ablock in blocks:
            self._cleanup_blocks.append(weakref.ref(ablock, self._cleanup_blocks.remove))

    def clean(self, bloc):
        """ remove markings from block.  if final is True
            remove from all register lists in the full stack"""
        for atom in bloc.atoms():
            self._unmark_atom(atom)


class ExposeOnly(Protect):
    """
    Opposite operation to Protect, exposing only the given atom
    indices to reaction
    """
    def __init__(self, *sites):
        Protect.__init__(self, *sites)
        sites_grouped_by_block = []
        for site in sites:
            found_match = False
            for site_list in sites_grouped_by_block:
                if site.block is site_list[0].block:
                    site_list.append(site)
                    found_match = True
            if not found_match:
                sites_grouped_by_block.append([site])

        self._sites = []
        for site_group in sites_grouped_by_block:
            atom_count = site_group[0].block.mol.GetNumAtoms()
            protect = set(range(atom_count))  # make set of all atom indices
            for site in site_group:
                protect.difference_update(site.atom_indices)  # pull out the ones from site
            self._sites.append(ReactiveSite(site_group[0].block, protect))
