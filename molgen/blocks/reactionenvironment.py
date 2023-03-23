import logging
import re
import subprocess

from rdkit.Chem.AllChem import ReactionFromSmarts

from . import block
from . import protect
from . import reactivesite

__reaction_env__ = None


def current_environment():
    global __reaction_env__
    if __reaction_env__ is None:
        raise Exception("No Reaction Environment Found")
    return __reaction_env__


class LazyMessage:
    """
    Lazy class used to prevent the evaluation of a smiles string for
    debug messages in the reaction methods.  getting smiles is slow
    so this prevents the call unless you are actually at log level debug

    To use this class, you also need to call the logging message correctly

    like this:
    logging.debug("Message: %s", LazyMessage(expensive_func, arg1, arg2))
                               ^ note you don't use the '%' here
    more info here:
    http://stackoverflow.com/questions/21377020/python-how-to-do-lazy-debug-logging
    """
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        try:
            return str(self.func(*self.args, **self.kwargs))
        except:
            return "Failed Loading Lazy Message"  # this is important for debug.  sometimes intermediate molecules cant be written to smiles.

class ReactionEnvironment:
    """
    Describes the type of reactions that occur between blocks.  Blocks keep a
    reference to this environment when they are reacted together
    Initialised with the smarts string to be used for the reaction.
    """
    def __init__(self, smarts, handles, estimate_func=None, estimate_sym_func=None):
        """
        Initialization. The smarts string sets the reaction that will
        take place and the handles list dictates the reactive atoms.
        estimate_func is a function used to estimate the number of products
        given the number of available sites to react in blocks
        """
        self.smarts = smarts
        self.handles = handles
        self._old_reaction_env = None

        handles_plus_markers = [r + "(:[0-9]+)?" for r in self.handles]
        handles = "|".join(handles_plus_markers)
        self.handle_finder = re.compile("\[(" + handles + ")\]")

        if estimate_func is not None and not callable(estimate_func):
            raise ValueError("estimate_func must be callable")
        self._estimate_func = estimate_func

        if estimate_sym_func is not None and not callable(estimate_sym_func):
            raise ValueError("estimate_func must be callable")
        self._estimate_sym_func = estimate_sym_func

    def react(self, *blocks):
        raise NotImplementedError

    def find_reactive_sites(self, block, reactant_index=0):
        raise NotImplementedError

    def estimate(self, *args):
        try:
            return self._estimate_func(*args)
        except TypeError:
            msg = "Failed estimate with estimate_func {}"
            logging.exception(msg.format(self._estimate_func))
            raise

    def estimate_sym(self, *args):
        try:
            return self._estimate_sym_func(*args)
        except TypeError:
            msg = "Failed estimate with estimate_sym_func {}"
            logging.exception(msg.format(self.estimate_sym_func))
            raise

    def clean_smiles(self, smiles, replacement='H'):
        """
        return a smiles string from given smiles string with
        reactive handles replaced with specified replacement.
        Default is Hydrogen
        """
        cleaned_smiles = self.handle_finder.sub('['+replacement+']', smiles)
        return cleaned_smiles

    def __enter__(self):
        global __reaction_env__
        if "__reaction_env__" in globals():
            self._old_reaction_env = __reaction_env__
        __reaction_env__ = self

    def __exit__(self, exception_type, exception_value, traceback):
        global __reaction_env__
        __reaction_env__ = self._old_reaction_env


class RDKitReactionEnvironment(ReactionEnvironment):
    """
    Describes the type of reactions that occur between blocks.  Blocks keep a
    reference to this environment when they are reacted together
    Initialised with the smarts string to be used for the reaction.
    """

    def __init__(self, smarts, handles, estimate_func=None):
        """
        Initialization. The smarts string sets the reaction that will take place
        and the handles list dictates the reactive atoms.
        estimate_func is a function used to estimate the number of products given
        input arguments matching a reaction
        """
        ReactionEnvironment.__init__(self, smarts, handles, estimate_func)
        if self.smarts:
            self.rxn = ReactionFromSmarts(self.smarts)
        else:
            self.rxn = None

    @protect.honor_protected_atoms
    def react(self, left_block, right_block):
        """
        Reacts the two molecules together and returns only the left side.  Currently takes smiles strings
        but this is likely to be converted to inchi in the near future for
        consistancy
        """
        if self.rxn is None:
            raise Exception("Unable to call react with no smarts")
        mol_1 = left_block.mol
        mol_2 = right_block.mol
        prods = self.rxn.RunReactants((mol_1, mol_2))
        # this is specific to our kind of smarts string where we only want left reactants.
        first_prods = set([prod[0] for prod in prods])
        blocks = [block.Block(mol=product) for product in first_prods]

        if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
            for b in blocks:
                logging.debug("%s + %s = %s",
                              LazyMessage(left_block.smiles, True),
                              LazyMessage(right_block.smiles, True),
                              LazyMessage(b.smiles, True))
        return blocks

    def find_reactive_sites(self, block, reactant_index=0):
        """
        return list of tuples providing atom numbers of reactive sites
        for the current reaction environment

        reactant index indicates the index of the reactant according to
        the current SMARTS string.  Default is 0, meaning the leftmost
        reactant
        """
        if self.rxn is None:
            raise Exception("Unable to find reactive sites with no smarts")
        reactant_num = self.rxn.GetNumReactantTemplates()
        if reactant_index >= reactant_num:
            msg = "reactant_index {} greater than number "
            msg += " of reactants ({}) in current reaction {}"
            raise ValueError(msg.format(reactant_index, reactant_num, self.smarts))

        reactant_template = self.rxn.GetReactantTemplate(reactant_index)
        atom_indices = block.mol.GetSubstructMatches(reactant_template)
        return [reactivesite.ReactiveSite(block, atuple) for atuple in atom_indices]

    def __str__(self):
        return "RDKitReactionEnvironment({})".format(self.smarts)


class ChemAxonReactionEnvironment(ReactionEnvironment):
    """
    A reaction environment that utilizes chemaxon's react
    via a subprocess call to react.  Expects SMILES

    WARNING: this Reaction Environment does not implement some
    methods so may raise errors in certain scenarios, use cautiously
    """

    def react(self, left_block, right_block):
        """
        return list of left side products from chemaxon's react
        """
        command = ['react',
                   '-m',
                   'comb',
                   '-r',
                   self.smarts,
                   left_block.smiles(),
                   right_block.smiles(),
                   '-f',
                   'smiles:u'
                   ]

        ret_string = subprocess.check_output(command)
        ret_list = ret_string.split()
        return ret_list[::2]

    def __str__(self):
        return "ChemAxonReactionEnvironment({})".format(self.smarts)
