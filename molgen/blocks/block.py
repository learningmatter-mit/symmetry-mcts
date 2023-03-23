import itertools
import logging
import re
from hashlib import sha512
from . import reactionenvironment
from . import protect

# try statement around import is a hack to allow ReadTheDocs to work
try:
    from rdkit.Chem import (MolToSmiles, MolFromSmiles, SanitizeMol,
                            MolFromInchi, MolToInchi, InchiToInchiKey, Mol)
except ImportError:
    print("You must install rdkit")


class Block:
    """
    The block is the local object that is used for storing and manipulating
    reactive molecules.  The key will be set to the inchi of the reactive
    molecule.  The block class method has inbuilt methods that allow the
    blocks to be reacted with other blocks and variables that take care of the
    book-keeping (children, birth_reaction and ancestors).
    In the context of a block, ancestors are the blocks which combined to
    produce this block, children are new blocks that were produced from this
    block, and the birth_reaction is the ReactionEnvironment that was used to
    produce the children.

    Using either *reconnect_metals* or *fixed_hydrogens* will produce NEW inchi keys
    that are scientifically more sound for organometallics and mobile H atoms BUT
    will result in different inchis and inchikeys even for the molecules that don't
    need the option on.
    """
    _marker_finder = re.compile("(:[0-9]+)?(?=\])")

    def __init__(self, smiles=None, inchi=None, mol=None):
        """
        Initialisation; the key is the inchi string that describes the molecule
        and the rxn_env is the reaction environment that will be used for
        combining blocks
        """

        if mol is None:
            if inchi is None and smiles is None:
                raise ValueError("You must specify either a smiles or an InChI")
            elif inchi is not None and smiles is not None:
                raise ValueError("You must specify either a smiles or an InchI")
            try:
                if smiles:
                    self.mol = MolFromSmiles(str(smiles))
                if inchi:
                    self.mol = MolFromInchi(str(inchi))
                if self.mol is None:
                    raise Exception("Mol not created")
            except:
                msg = "Could not create RdKit mol from smiles: {}, inchi: {}"
                logging.exception(msg.format(smiles, inchi))
                raise
        else:
            self.mol = mol
        SanitizeMol(self.mol)

        self.inchi_option_string = " -RecMet  -FixedH "

        _nonstd_inchi_string = MolToInchi(self.mol,
                                          options=self.inchi_option_string)

        self.inchikey = InchiToInchiKey(_nonstd_inchi_string)
        self._inchi_string = _nonstd_inchi_string

        self.parents = []
        self.birth_reaction = None
        self.birth_method = None
        self.generation = 0
        self.ancestors = []

    def smiles(self, show_markers=True, smiles_stereo=True):
        """
        get smiles string.  if show markers is False
        strip out any markers such that [Mg:100] becomes [Mg]
        """

        smiles = MolToSmiles(self.mol, isomericSmiles=smiles_stereo)
        if show_markers:
            return smiles
        else:
            new_mol = MolFromSmiles(self._marker_finder.sub('', smiles))
            new_smiles = MolToSmiles(new_mol, isomericSmiles=smiles_stereo)
            return new_smiles

    def add_ancestors(self, ancestors):
        """
        Set the ancestors, i.e. the reactive molecules that combined to make
        this block.  Takes an iterable of Blocks.
        """
        for ancestor in ancestors:
            if not isinstance(ancestor, Block):
                raise ValueError("Ancestor must be of type Block")
        for ancestor in ancestors:
            if ancestor not in self.ancestors:
                self.ancestors.extend(ancestors)

    def set_parents(self, parents):
        """
        Set the ancestors, i.e. the reactive molecules that combined to make
        this block.  Takes an iterable of Blocks.
        """
        for parent in parents:
            if not isinstance(parent, Block):
                raise ValueError("Parent must be of type Block.  received {}".format(parent))
        for p in parents:
            self.add_ancestors(p.ancestors)
        self.parents = list(parents)

    def _add_parents(self, parents):
        """
        add the parent blocks provided to
        this block.  Takes an iterable of Blocks.

        this doesn't overwrite existing parents, but extends list
        """
        for parent in parents:
            if not isinstance(parent, Block):
                raise ValueError("Parent must be of type Block.  received {}".format(parent))
        for p in parents:
            self.add_ancestors(p.ancestors)
            if p not in self.parents:
                self.parents.append(p)

    def take_heritage(self, other):
        self.set_parents(other.parents)
        self.birth_reaction = other.birth_reaction
        self.birth_method = other.birth_method
        self.generation = other.generation

    def copy(self, strip_markers=False):
        copyblock = Block(mol=Mol(self.mol))
        copyblock.take_heritage(self)
        if strip_markers:
            for a in copyblock.atoms():
                a.ClearProp(protect.PROTECT_PROP_KEY)
        return copyblock

    def _copy_duplicates(self, blocks):
        """
        check blocks for duplicates to self and for
        duplicates within eachother.  If duplication
        occurs, replace one of the duplicates with a copy.
        """
        block_list = list(blocks)
        # first check for self in block_list
        for j, other_block in enumerate(block_list):
            if self is other_block:
                block_list[j] = other_block.copy()
        # now check for duplicates within the block_list
        for i, a_block in enumerate(block_list):
            for j, other_block in enumerate(block_list):
                if i != j and a_block is other_block:
                    block_list[i] = a_block.copy()
        return tuple(block_list)

    def _react_block(self, *others, **kwargs):
        merge_parents = kwargs.get("merge_parents", False)
        rxn_env = reactionenvironment.current_environment()
        react_string = str(rxn_env)
        others = self._copy_duplicates(others)
        all_blocks = (self,) + others
        new_parents = [a.copy(strip_markers=True) for a in all_blocks]
        for baby in set(rxn_env.react(*all_blocks)):  # wrap products in set to reduce dupes
            if merge_parents:
                baby.set_parents(self.parents)
                baby._add_parents(new_parents[1:])  # don't add self
                baby.generation = self.generation
            else:
                baby.set_parents(new_parents)
                baby.generation = self.generation + 1
            baby.birth_reaction = react_string
            baby.birth_method = "react"
            yield baby

    def react(self, *others, **kwargs):
        """
        Generator that yields blocks

        React self with n other building blocks.
        Uses the current reaction environment for reaction.
        Each product will be a permutation of a single reaction
        site from self and a single reaction site from others
        """
        if all([hasattr(o, '__iter__') for o in others]):
            for other_blocks in itertools.product(*others):
                for prod in self._react_block(*other_blocks, **kwargs):
                    yield prod
        else:
            for prod in self._react_block(*others, **kwargs):
                yield prod

    def react_sym(self, *others):
        """
        Generator that yields product blocks with all reactive sites
        in block self filled symmetrically with all permuations of others

        it's important to note symmetrical nature of the output.
        All products will have sites of self full, but we can still permute
        the orientation (and combinations if more than one other reactant)
        of the right side reactants, but only in a way that the results are
        symmetric.  If we are reacting block B and block P, where B
        has two reactive sites and is symmetrical and where P has two
        reactive sites and is not symametrical, so it can be connected to B
        in two ways P1, and P2.  react_sym will return: P1-B-P1 and P2-B-P2.
        It will not return P1-B-P2.
        """
        if all([hasattr(o, '__iter__') for o in others]):
            for other_blocks in itertools.product(*others):
                for prod in self._react_sym_block(*other_blocks):
                    yield prod
        else:
            for prod in self._react_sym_block(*others):
                yield prod

    def _react_sym_block(self, *others):
        self.my_site_count = len(self.sites())
        other_sites = []
        others = self._copy_duplicates(others)
        for i, other in enumerate(others):
            # get sites for the (i+1)th reactant in the reactant list
            # of the current reaction environment
            # self will get reactant 0, thus +1
            site_list = other.sites(i+1)
            other_sites.append(site_list)
        parents = (self,) + others
        for one_site_each_from_others in itertools.product(*other_sites):
            if self.my_site_count > 0:
                product = self
                with protect.ExposeOnly(*one_site_each_from_others):
                    for i in range(self.my_site_count):
                        products = tuple(product.react(*others))
                        if products:
                            product = products[0]
                product.set_parents(parents)
                product.generation = self.generation + 1
                product.birth_method = "react_sym"
                yield product

    def react_fill(self, *others):
        """
        recursive generator that yields product blocks with all reactive sites
        in block self filled with all permuations of others

        it's important to note asymmetry  of the output.
        look to react_sym for the symmetrical version of react_fill
        """
        for prod in self._react_all(*others, yield_incomplete_blocks=False, topcall=True):
            prod.birth_method = "react_fill"
            yield prod

    def react_all(self, *others):
        """
        recursive generator that yields product blocks with all reactive sites
        in block self filled with all permuations of others including not reacting
        at a site.

        it's important to note asymmetry of the output.
        look to react_sym for symmetrical output.
        It is also important to note that this will produce products with reactive
        sites remaining.

        """
        for prod in self._react_all(*others, yield_incomplete_blocks=True, topcall=True):
            prod.birth_method = "react_all"
            yield prod

    def _react_all(self, *others, **kwargs):
        """
        A private inner reactor that can do either react_fill or react_all depending
        on the keyword arg passed in: "yield_incomplete_blocks"
        if True, it will be react_all, if False, it's react_fill
        """
        self.my_site_count = len(self.sites(0, unprotected_only=True))
        if self.my_site_count == 0:
            # base case
            # in the case that you are yielding blocks on the way to making the final
            # don't both yielding self again.  otherwise yield self, BUT ONLY if you are not the top level
            # In the case of the top level. _react_all simply has no results
            if not kwargs["yield_incomplete_blocks"] and not kwargs["topcall"]:
                yield self
        else:
            is_top_call = kwargs["topcall"]
            kwargs["topcall"] = False
            if all([hasattr(o, '__iter__') for o in others]):
                set_of_others = others
            else:
                set_of_others = [[o] for o in others]
            # check for self in others and duplicate it
            set_of_others = [[o if o is not self else o.copy() for o in others] for others in set_of_others]


            for other_blocks in itertools.product(*set_of_others):
                other_sites = []
                #other_blocks = self._copy_duplicates(other_blocks)
                # make a local copy to use for protecting and reacting against
                # keep an unprotected copy locally to use in recursive case
                local_others = tuple(o.copy() for o in other_blocks)
                for i, other in enumerate(local_others):
                    # get sites for the (i+1)th reactant in the reactant list
                    # of the current reaction environment
                    # self will get reactant 0, thus +1
                    site_list = other.sites(i+1, unprotected_only=True)
                    other_sites.append(site_list)
                list_of_site_options = itertools.product(*other_sites)
                for one_site_each_from_others in list_of_site_options:
                    with protect.ExposeOnly(*one_site_each_from_others) as protect_other:
                        # only react and recurse with products of reacting a single site on self
                        if kwargs["yield_incomplete_blocks"]:
                            explore_site_count = len(self.sites(0, unprotected_only=True))
                        else:
                            explore_site_count = 1

                        for left_site_index in range(explore_site_count):
                            products = set()
                            with protect.ExposeOnly(self.sites(0, unprotected_only=True)[left_site_index]):
                                products |= set(self.react(*local_others, merge_parents=not is_top_call))
                            # leave protect stack since we don't want to keep that protection
                            with protect.Protect(self.sites(0, unprotected_only=True)[left_site_index]):
                                for prod in products:
                                    if kwargs["yield_incomplete_blocks"]:
                                        # this is the only line diff from react_fill
                                        outcopy = prod.copy(strip_markers=True)
                                        yield outcopy
                                    for inner_prod in prod._react_all(*set_of_others, **kwargs):
                                        protect_other.clean(inner_prod)
                                        yield inner_prod

                        # # in the react_all case, now also react the remaining sites on
                        # # self and return those as well.
                        # if kwargs["yield_incomplete_blocks"]:
                        #     with protect.Protect(self.sites(0)[0]) as protect_self:
                        #         for p in self.react(*local_others, merge_parents=not is_top_call):
                        #             protect_self.clean(p)
                        #             protect_other.clean(p)
                        #             yield p

    def inchi(self):
        """return inchi string for this block"""
        return self._inchi_string

    def cleaned(self, replacement='H', clean_parents=True):
        """
        return a new Block that is a version of this block with all
        reactive handles (according to current reaction environment)
        replaced with given replacement.  default: Hydrogen
        """
        rxn_env = reactionenvironment.current_environment()
        cleaned = rxn_env.clean_smiles(self.smiles(), replacement)
        cb = Block(smiles=cleaned)
        cb.take_heritage(self)
        if clean_parents:
            cb.parents = [p.cleaned(replacement, clean_parents) for p in cb.parents]
        return cb

    def estimate(self, *others):
        """
        estimate the product count given other reactants
        """
        if all([hasattr(o, '__iter__') for o in others]):
            return self._estimate_blockset(*others)
        else:
            rxn_env = reactionenvironment.current_environment()
            all_blocks = (self,) + others
            site_counts = [len(b.sites(i)) for i, b in enumerate(all_blocks)]
            return rxn_env.estimate(*site_counts)

    def _estimate_blockset(self, *others):
        my_site_count = len(self.sites(0))
        total_site_count_list = [my_site_count]
        for i, bset in enumerate(others):
            total = sum([len(o.sites(i+1)) for o in bset])
            total_site_count_list.append(total)
        rxn_env = reactionenvironment.current_environment()
        return rxn_env.estimate(*total_site_count_list)

    def estimate_sym(self, *others):
        """
        estimate the symmetrical product count given other reactants
        """
        if all([hasattr(o, '__iter__') for o in others]):
            return self._estimate_sym_blockset(*others)
        else:
            rxn_env = reactionenvironment.current_environment()
            all_blocks = (self,) + others
            site_counts = [len(b.sites(i)) for i, b in enumerate(all_blocks)]
            return rxn_env.estimate_sym(*site_counts)

    def _estimate_sym_blockset(self, *others):
        total = 0
        for other_blocks in itertools.product(*others):
            total += self.estimate_sym(*other_blocks)
        return total

    def sites(self, reactant_index=0, unprotected_only=False):
        """
        return list of tuples providing atom numbers of reactive sites
        for the current reaction environment

        reactant index indicates the index of the reactant according to
        the current SMARTS string.  Default is 0, meaning the leftmost
        reactant
        """
        rxn_env = reactionenvironment.current_environment()
        all_sites = rxn_env.find_reactive_sites(self, reactant_index)
        if unprotected_only:
            return [s for s in all_sites if not s.is_protected()]
        else:
            return all_sites

    def atoms(self):
        """return a list of rdkit Atom objects for this molecule"""
        return [self.mol.GetAtomWithIdx(i) for i in range(self.mol.GetNumAtoms())]

    def __eq__(self, other):
        """ overload equality based on inchikey """
        try:
            return self.inchikey == other.inchikey
        except AttributeError:
            return False

    def __ne__(self, other):
        try:
            return self.inchikey != other.inchikey
        except AttributeError:
            return True

    def __cmp__(self, other):
        return cmp(self.smiles(), other.smiles())

    def __hash__(self):
        """
        return integer by hashing the inchikey, which is in-turn a hash!
        this method allows blocks to be in sets
        """
        return int(sha512(self.inchikey.encode('utf-8')).hexdigest(), 16)

    def __repr__(self):
        return "<Block {}>".format(self.smiles())
