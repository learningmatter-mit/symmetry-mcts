from molgen.blocks import Block

class MolStorage:
    def __init__(self, smiles, label, group, blocks):
        self.smiles = smiles
        self.label = label
        self.group = group
        self.blocks = blocks

    def get_blocks(self, require_symmetric=False, 
                output_format=None):
        for b in self.blocks:
            if not require_symmetric or b["symmetric"]:
                if output_format == 'raw_smiles':
                    out = b['smiles']
                else:
                    out = Block(smiles=b["smiles"])
                    out.label = self.label
                yield out

