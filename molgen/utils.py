from pgmols.models import Mol, Group, Species, MolSet, Stoichiometry, MolSet, Cluster
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import SanitizeFlags
from rdkit.Chem import SanitizeMol
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.rdmolops import GetFormalCharge
from rdkit.Chem import SanitizeFlags
from rdkit.Chem import SanitizeMol
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.rdmolops import GetFormalCharge

from collections import Counter
import itertools
from confgen.stereochem_utils import PartialSanitizeMolFromSmiles
from rdkit import RDLogger
from django.db import IntegrityError

INCHI_OPTION_STRING = " -RecMet  -FixedH "


def put_block(block, project, tags=[], details={}, return_mol=False, multiplicity=1):
    '''
    A method for loading a block into a database
    '''
    return put_mol(
        rdkit_mol=block.mol,
        project=project,
        tags=tags,
        details={**{'generation': block.generation}, **details},
        parent_inchikeys=[parent.inchikey for parent in block.parents],
        calc_charge=True,
        return_mol=return_mol,
        inchikey=block.inchikey,
        multiplicity=multiplicity
    )


def put_mol(rdkit_mol=None, project=None, tags=[], details={},
            parent_inchikeys=[], calc_charge=True, return_mol=False,
            inchikey=None, species_only=False, multiplicity=1):
    '''
    Every molecule (graph that comes out of molgen) has an attached species,
    which can be a cluster of other species, and has a multiplicity and a stoichimetry.
    Anything can be a species as long as it has a SMILES
    A short-lived intermediate or even a transition states.
    '''

    group, group_created = Group.objects.get_or_create(name=project)

    formula = rdMolDescriptors.CalcMolFormula(rdkit_mol)
    pgstoich = Stoichiometry.objects.filter(formula=formula).first()
    mass = MolWt(rdkit_mol)
    charge = GetFormalCharge(rdkit_mol)

    if pgstoich is None:
        pgstoich = Stoichiometry(formula=formula,
                                 mass=mass,
                                 charge=charge)
        pgstoich.save()

    if not inchikey:
        # if flex_val is true, don't sanitize since molecule will have already
        # been partially sanitized and rezanitizaiton would break the code
        if details.get('flex_val', False):
            pass
        else:
            Chem.SanitizeMol(rdkit_mol)
        non_std_inchi = Chem.MolToInchi(rdkit_mol,
                                        options=str(INCHI_OPTION_STRING))
        inchikey = Chem.InchiToInchiKey(non_std_inchi)
    smiles = Chem.MolToSmiles(rdkit_mol)

    # ref for get or create
    # https://kite.com/python/docs/django.db.models.QuerySet.get_or_create
    try:
        pgspecies, new_species = Species.objects.get_or_create(
            defaults={'smiles': smiles,
                      'inchikey': inchikey,
                      'details': details},
            # details is now passed in to make sure that "get" doesn't find an existing molecule,
            # create places the details field inside
            # it's important to know that defaults is only used in the create method
            # of the get_or_create function.
            smiles=smiles,
            group=group,
            stoichiometry=pgstoich,
            inchikey=inchikey,
            multiplicity=multiplicity)
    except IntegrityError:
        pgspecies = Species.objects.get(group=group,
                                        stoichiometry=pgstoich,
                                        inchikey=inchikey,
                                        multiplicity=multiplicity)
        new_species = False
        print("Species already exists with SMILES "
              "{} new SMILES {} has same inchikey".format(pgspecies.smiles,
                                                          smiles))

        newdetails = {}
        if pgspecies.details is not None:
            newdetails = pgspecies.details
        if details:
            newdetails = {**newdetails, **pgspecies.details}
        pgspecies.details = newdetails
        pgspecies.save()

    if "." in smiles:
        species_only = True

    if species_only:
        if return_mol:
            return {'species': pgspecies, 'created': new_species}
        else:
            return new_species

    try:
        mol, mol_created = Mol.objects.get_or_create(
            defaults={'smiles': smiles,
                      'inchikey': inchikey,
                      'details': details},
            inchikey=inchikey,
            species=pgspecies,
            group=group)
    except IntegrityError:
        mol = Mol.objects.get(species=pgspecies,
                              group=group,
                              inchikey=inchikey)
        mol_created = False
        self.stdout.write("Mol already exists with SMILES "
                          "{} new SMILES {} has same inchikey".format(mol.smiles,
                                                                      smiles))

    for tag in tags:
        pg_molset, new_molset = MolSet.objects.get_or_create(group=group,
                                                             name=tag)
        pg_molset.mols.add(mol)

    for parent_inchikey in parent_inchikeys:
        try:
            parent_mol = Mol.objects.get(group=group,
                                         inchikey=parent_inchikey)
            mol.parents.add(parent_mol)
        except Mol.DoesNotExist:
            pass

    mol.save()
    if return_mol:
        return {'mol': mol, 'created': mol_created}
    else:
        return mol_created


def get_molecular_charge_props(rdkit_mol):
    molecular_charge = Chem.GetFormalCharge(rdkit_mol)
    num_protons = sum([a.GetAtomicNum()
                       for a in Chem.AddHs(rdkit_mol).GetAtoms()])
    electron_count = num_protons + molecular_charge
    return {'molecular_charge': molecular_charge,
            'electron_count': electron_count}


def put_smiles(smiles=None, **kwargs):
    return put_mol(rdkit_mol=Chem.MolFromSmiles(smiles), **kwargs)


def enumerate_stereoisomers(smiles):
    opts = Chem.StereoEnumerationOptions(tryEmbedding=True)
    m = Chem.MolFromSmiles(smiles)
    isomers = [Chem.MolToSmiles(i) for i in tuple(Chem.EnumerateStereoisomers(m,
                                                  options=opts))]
    return isomers


def breakdown_clusters(species, smiles, moltags=[]):
    allparents = []
    if species.details:
        speciesdetails = species.details
    else:
        speciesdetails = {}
    for component in smiles.split("."):
        parentformula, parentmass, parentcharge, parentsmiles, parentinchikey = process_smiles(
            component, speciesdetails.get('flex_val', False))

        pgstoich = Stoichiometry.objects.filter(
            formula=parentformula).first()

        if pgstoich is None:
            pgstoich = Stoichiometry(formula=parentformula,
                                     mass=parentmass,
                                     charge=parentcharge)
            pgstoich.save()

        try:
            pspec, new_sp = Species.objects.get_or_create(smiles=parentsmiles,
                                                          group=species.group,
                                                          stoichiometry=pgstoich,
                                                          inchikey=parentinchikey,
                                                          multiplicity=1)
        except IntegrityError:
            pspec = Species.objects.get(group=species.group,
                                        stoichiometry=pgstoich,
                                        inchikey=parentinchikey,
                                        multiplicity=1)
            print("Species already exists with SMILES "
                  "{} new SMILES {} has same inchikey".format(pspec.smiles, parentsmiles))

        try:
            parentpgmol, new_m = Mol.objects.get_or_create(smiles=parentsmiles,
                                                           species=pspec,
                                                           group=species.group,
                                                           inchikey=parentinchikey)
        except IntegrityError:
            parentpgmol = Mol.objects.get(species=pspec,
                                          group=species.group,
                                          inchikey=parentinchikey)
            new_m = False

            print("Mol already exists with SMILES "
                  "{} new SMILES {} has same inchikey".format(parentpgmol.smiles,
                                                              parentsmiles))

        parentpgmol.tag(moltags)
        allparents.append(pspec)

    combinerecursive(allparents)


def combinerecursive(pgspclist):
    allclusters = set()
    for combsize in range(2, len(pgspclist) + 1):
        for per in set(itertools.combinations(pgspclist, combsize)):
            largestcluster = per[0]
            allclusters.add(largestcluster)
            for i in range(len(per) - 1):
                largestcluster = combine_species(largestcluster,
                                                 per[i + 1],
                                                 update_existing=True
                                                 )
                allclusters.add(largestcluster)
    return allclusters


def process_smiles(smiles,
                   flex_val=False):
    if not flex_val:
        rdmol = Chem.MolFromSmiles(smiles)
    else:
        rdmol = Chem.MolFromSmiles(str(smiles), sanitize=False)
        rdmol.UpdatePropertyCache(strict=False)
        allflags = SanitizeFlags.SANITIZE_FINDRADICALS
        allflags |= SanitizeFlags.SANITIZE_KEKULIZE
        allflags |= SanitizeFlags.SANITIZE_SETAROMATICITY
        allflags |= SanitizeFlags.SANITIZE_SETCONJUGATION
        allflags |= SanitizeFlags.SANITIZE_SETHYBRIDIZATION
        allflags |= SanitizeFlags.SANITIZE_SYMMRINGS
        SanitizeMol(rdmol, allflags, catchErrors=True)
    formula = rdMolDescriptors.CalcMolFormula(rdmol)
    mass = MolWt(rdmol)
    charge = GetFormalCharge(rdmol)
    smiles = Chem.MolToSmiles(rdmol)
    non_std_inchi = Chem.MolToInchi(rdmol,
                                    options=str(INCHI_OPTION_STRING))
    inchikey = Chem.InchiToInchiKey(non_std_inchi)
    return formula, mass, charge, smiles, inchikey


def get_inchi(smiles):
    rdmol = Chem.MolFromSmiles(str(smiles), sanitize=False)
    rdmol.UpdatePropertyCache(strict=False)
    allflags = SanitizeFlags.SANITIZE_FINDRADICALS
    allflags |= SanitizeFlags.SANITIZE_KEKULIZE
    allflags |= SanitizeFlags.SANITIZE_SETAROMATICITY
    allflags |= SanitizeFlags.SANITIZE_SETCONJUGATION
    allflags |= SanitizeFlags.SANITIZE_SETHYBRIDIZATION
    allflags |= SanitizeFlags.SANITIZE_SYMMRINGS
    SanitizeMol(rdmol, allflags, catchErrors=True)

    return Chem.MolToInchi(rdmol, options=str(INCHI_OPTION_STRING))


def combine_species(species1, species2, update_existing=False):
    full_smiles = species1.smiles + '.' + species2.smiles

    rdkit_mol = None
    details = {}
    # If either of the two species has flexible valency, so will the cluster of the two.

    if species2.details is not None:  # checks if species2 contains flex_val
        if species2.details.get('flex_val', False):
            rdkit_mol = PartialSanitizeMolFromSmiles(full_smiles)
            details = {'flex_val': True}
        else:
            rdkit_mol = Chem.MolFromSmiles(full_smiles)
            details = {}
    else:
        rdkit_mol = Chem.MolFromSmiles(full_smiles)
        details = {}

    # checks if species 1 has flex_val and only executes below if species 2 didn't have it
    if species1.details is not None and not details:
        if species1.details.get('flex_val', False):
            rdkit_mol = PartialSanitizeMolFromSmiles(full_smiles)
            details = {'flex_val': True}
        else:
            rdkit_mol = Chem.MolFromSmiles(full_smiles)
            details = {}

    elif not details:  # if neither of the two had details or flex_val, process details and rdkit_mol normally
        rdkit_mol = Chem.MolFromSmiles(full_smiles)
        details = {}

    assert(species1.group == species2.group)
    project = species1.group.name

    dictio = put_mol(rdkit_mol=rdkit_mol,
                     project=project,
                     tags=[],
                     details=details,
                     parent_inchikeys=[],
                     calc_charge=True,
                     return_mol=True,
                     inchikey=None,
                     species_only=True)

    pgspecies = dictio['species']
    if dictio['created'] is True or update_existing:
        if pgspecies.components.exists():
            return pgspecies
        base_species = []
        spec1_connections = Cluster.objects.filter(
            superspecies=species1).distinct()
        spec2_connections = Cluster.objects.filter(
            superspecies=species2).distinct()
        if spec1_connections.count() == 0:
            base_species += [species1]
        else:
            base_species += [
                i.subspecies for i in spec1_connections for j in range(i.subcount)]
        if spec2_connections.count() == 0:
            base_species += [species2]
        else:
            base_species += [
                i.subspecies for i in spec2_connections for j in range(i.subcount)]

        for spec, count in Counter(base_species).items():
            cluster = Cluster(superspecies=dictio['species'],
                              subspecies=spec,
                              subcount=count)
            cluster.save()

        pgspecies.componentcount = len(base_species)

    pgspecies.save()

    return pgspecies
