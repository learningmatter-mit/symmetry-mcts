import os
import subprocess
import sys
import logging
import functools

from rdkit.Chem import Draw

MAX_ATOM_COUNT = 1000


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    chunksize = int(lv / 3)
    return tuple(int(value[i:i+chunksize], 16)/255. for i in range(0, lv, chunksize))


left = "[Fe]C#N"
right = "Cc1ccc2c(c1)c1cc(C)ccc1n2"
prod = "Cc1ccc2c(c1)c1cc(C)ccc1n2C#N"
RED = [v/255. for v in (239, 138, 98)]
BLUE = [v/255. for v in (103, 169, 207)]
BLACK = (0, 0, 0)

if sys.platform == "darwin":
    CHEMAXON_BIN = "/Applications/ChemAxon/JChem/bin/"
else:
    CHEMAXON_BIN = "/opt/ChemAxon/JChem/bin/"


class MyDrawingOptions(object):
    dotsPerAngstrom = 30
    useFraction = 0.85

    atomLabelFontFace = "sans"
    atomLabelFontSize = 18
    atomLabelMinFontSize = 10

    bondLineWidth = 1
    dblBondOffset = .3
    dblBondLengthFrac = .8

    defaultColor = (1, 0, 0)
    selectColor = (1, 0, 0)

    colorBonds = True
    noCarbonSymbols = True
    includeAtomNumbers = False
    atomNumberOffset = 0
    radicalSymbol = u'\u2219'

    dash = (4, 4)

    wedgeDashedBonds = True

    # used to adjust overall scaling for molecules that have been laid out with non-standard
    # bond lengths
    coordScale = 1

    # elemDict={
    #    1:(0.55,0.55,0.55), # H
    #    7:(0,0,1),    # N
    #   8:(1,0,0),    # O
    #   9:(.2,.8,.8),  # F
    #   15:(1,.5,0),   # P
    #   16:(.8,.8,0),  # S
    #   17:(0,.8,0),   # Cl
    #   35:(.5,.3,.1), # Br
    #   0:(.5,.5,.5),
    #   }
    # colors taken from http://en.wikipedia.org/wiki/CPK_coloring
    elemDict = {1: (0.55, 0.55, 0.55),        # H
                7: hex_to_rgb("#2233ff"),     # N
                8: hex_to_rgb("#ff2200"),     # O
                9: hex_to_rgb("#55bb00"),     # F
                15: hex_to_rgb("#ff9900"),    # P
                16: hex_to_rgb("#bbaa00"),    # S
                17: hex_to_rgb("#55bb00"),    # Cl
                35: hex_to_rgb("#992200"),    # Br
                0: (.5, .5, .5),
                }


def _prep_file(block, path, filename, force):
    if filename is None:
        filename = block.inchikey + ".svg"
    filepath = os.path.join(path, filename)
    file_exists = os.path.isfile(filepath)
    if force and file_exists:
        os.remove(filepath)
    if not file_exists and not os.path.isdir(path):
        os.makedirs(path)
    return filepath, file_exists


def write_svg(block,
              path=".",
              filename=None,
              force=False,
              fontsize=0.8,
              groups=[]):
    """
    write svg file using molconvert for a block to the given path
    if filename is not given, use the inchikey + .svg as the filename
    force will force overwrite, otherwise just return path to existing file
    groups is a list of strings such as ['SO3H','COOH','PO3H2'] which
    will be rendered in compact form
    """

    try:
        filepath, file_exists = _prep_file(block, path, filename, force)
        if not file_exists or force:
            if groups:
                cmd = [CHEMAXON_BIN+"standardize"]
                if groups:
                    cmd += ["-c", 'creategroup:' + ','.join(groups)]
                cmd += [block.smiles(), '--output', filepath, "--format", "svg"]
                logging.debug(' '.join(cmd))
                subprocess.check_output(cmd, shell=False)
            else:
                cmd = "{}molconvert svg:atsiz{},transbg,cv_off -s '{}' -o {} -2"
                cmd = cmd.format(CHEMAXON_BIN, fontsize, block.smiles(), filepath)
                logging.debug(cmd)
                subprocess.check_output(cmd, shell=True)

        return filepath
    except:
        logging.exception("standardize SVG write failed with {}".format(block))
        raise


def by_size(a, b):
    """ given two tuples with first item being a block, sort by atom count"""
    a = len(a[0].atoms())
    b = len(b[0].atoms())
    return (a > b) - (a < b)


def write_svg_rdkit(block,
                    path=".",
                    filename=None,
                    force=False,
                    size=300,
                    colorize_substructures=[]):
    """
    write svg file for a block to the given path
    if filename is not given, use the inchikey + .svg as the filename
    force will force overwrite, otherwise just return path to existing file
    fontsize, size, and colorize_substructure tune the output

    colorize_substructures is a list of tuples of form (block, rgb) where rgb is a of form (0.5, 0.5, 0.5)

    """

    try:
        filepath, file_exists = _prep_file(block, path, filename, force)
        if not file_exists or force:
            mol = block.mol
            cmap = dict.fromkeys(range(MAX_ATOM_COUNT), BLACK)
            # make sure to colorize the smaller side first
            # this ensures that a sub-block doesn't overwrite a larger one.
            colorize_substructures.sort(key=functools.cmp_to_key(by_size))

            for sub_block, color in colorize_substructures:
                if sub_block is not None:
                    atoms = mol.GetSubstructMatches(sub_block.mol)
                    atoms = [element for tupl in atoms for element in tupl]
                    cmap.update(dict.fromkeys(atoms, color))
            if 0 not in cmap and 1 in cmap:
                cmap[0] = cmap[1]

            draw_options = MyDrawingOptions()
            if size < 200:
                draw_options.atomLabelFontSize = 1
                draw_options.atomLabelMinFontSize = 1
                draw_options.bondLineWidth = 1
            Draw.MolToFile(mol, filepath, size=(size, size), highlightMap=cmap, options=draw_options)
        return filepath
    except:
        logging.exception("rdkit SVG write failed with {}".format(block))
        raise


if __name__ == "__main__":
    from blocks import block
    import argparse

    import csv
    parser = argparse.ArgumentParser(description="Make SVGs")
    parser.add_argument('--csv', dest='csv', type=os.path.expanduser, required=True,
                        help="path to csv")
    parser.add_argument('--outpath', dest='path', type=os.path.expanduser, required=True,
                        help="output path for svgs")

    clargs = parser.parse_args()
    with open(clargs.csv) as fp:
        for m in csv.DictReader(fp):
            if "blas3" in m["tags"]:
                b = block.Block(smiles=m["smiles"])
                write_svg(b, path=clargs.path, molconvert=True)
