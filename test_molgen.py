from molgen import react

if __name__ == "__main__":
    # core = {
    # 		  "smiles": "c1cc2sc3c4c5nsnc5c6c7sc8ccsc8c7nc6c4nc3c2s1",
    # 		  "label": "opd",
    # 		  "group": "core",
    # 		  "blocks": [
    # 	{"smiles": "c1([He])c([Ne])c2sc3c4c5nsnc5c6c7sc8c([Ne])c([He])sc8c7n([Ar])c6c4n([Ar])c3c2s1"}
    # 	]
    # 	}

    # func_g = {
    # 		"smiles": "C=Cc1c(=C(C#N)C#N)c2cc([Cl])c([Cl])cc2c1(=o)",
    # 		"label": "opd",
    # 		"group": "end_group",
    # 		"blocks": [
    # 	{"smiles": "C([He])=C1C(=C(C#N)C#N)c2cc([Cl])c([Cl])cc2C1(=O)"}
    # 	]
    # 	}

    # print(react.run_sym('opd', core=core, functional_group=func_g, reactive_pos=0, pair_tuple=("a", "a")))

    # core = {
    # 		"smiles": "c1cc2sc3c4c5nnnc5c6c7sc8ccsc8c7nc6c4nc3c2s1",
    # 		"label": "opd",
    # 		"group": "core",
    # 		"blocks": [
    # 	{"smiles": "c1([He])c([Ne])c2sc3c4c5nn([Xe])nc5c6c7sc8c([Ne])c([He])sc8c7n([Ar])c6c4n([Ar])c3c2s1"}
    # 	]
    # 	}

    # core = {
    #     "smiles": "CCCCCCCCC(CCCCCC)Cc1csc2c1oc1c2sc2c3sc4c(oc5c(CC(CCCCCC)CCCCCCCC)csc54)c3c(F)c(F)c12",
    #     "label": "opd",
    #     "group": "core",
    #     "blocks": [
    #         {"smiles": "CCCCCCCCC(CCCCCC)Cc1c([He])sc2c1oc1c2sc2c3sc4c(oc5c(CC(CCCCCC)CCCCCCCC)c([He])sc54)c3c(F)c(F)c12"}
    # 	]
    # }

    core = {
        "smiles": "[H]c1ccc(-c2sc3c(oc4c3sc3c5sc6c(oc7c(CC(CCCCCC)CCCCCCCC)c(-c8ccc([H])s8)sc76)c5c(F)c(F)c43)c2CC(CCCCCC)CCCCCCCC)s1",
        "label": "opd",
        "group": "zero",
        "blocks": [
            {
                "smiles": "[He]c1ccc(-c2sc3c(oc4c3sc3c5sc6c(oc7c(CC(CCCCCC)CCCCCCCC)c(-c8ccc([He])s8)sc76)c5c(F)c(F)c43)c2CC(CCCCCC)CCCCCCCC)s1"
            }
        ],
    }

    # func_g = {
    #     "smiles": 'C=c1sccc1',
    #     "label": "opd",
    #     "group": "side_chain_1",
    #     "blocks": [
    #         {"smiles": "c([He])1sc([He])cc1"}
    # 	]
    # }

    func_g = {
        "smiles": "C=CC=C",
        "label": "opd",
        "group": "pi_bridge",
        "blocks": [{"smiles": "C([He])=CC=C([He])"}],
    }

    # func_g = {
    # 		"smiles": "CCCCCCCCCCC",
    # 		"label": "opd",
    # 		"group": "side_chain_1",
    # 		"blocks": [
    # 	{"smiles": "C([Xe])CCCCCCCCCC"}
    # 	]
    # 	}

    print(
        react.run(
            "opd",
            core=core,
            functional_group=func_g,
            reactive_pos=0,
            pair_tuple=("a", "a"),
        )
    )
# Expected Output:
# [
# 	{
# 		'smiles': 'N#Cc1cc(-c2c3ccccc3cc3cc4c(-c5cc(C#N)cc6cccnc56)c5ccccc5cc4cc23)c2ncccc2c1',
# 		'label': 'opd',
# 		'group': 'functionalized_core',
# 		'blocks': [
# 					{'smiles': '[He]c1ccc2cc(C#N)cc(-c3c4ccccc4cc4cc5c(-c6cc(C#N)cc7ccc([He])nc67)c6ccccc6cc5cc34)c2n1'},
# 					{'smiles': '[He]c1ccc2c(-c3cc(C#N)cc4cccnc34)c3cc4cc5cc([He])ccc5c(-c5cc(C#N)cc6cccnc56)c4cc3cc2c1'},
# 					{'smiles': '[He]c1cccc2c(-c3cc(C#N)cc4cccnc34)c3cc4cc5c([He])cccc5c(-c5cc(C#N)cc6cccnc56)c4cc3cc12'},
# 					{'smiles': '[He]c1cnc2c(-c3c4ccccc4cc4cc5c(-c6cc(C#N)cc7cc([He])cnc67)c6ccccc6cc5cc34)cc(C#N)cc2c1'},
# 					{'smiles': '[He]c1c(C#N)cc2cccnc2c1-c1c2ccccc2cc2cc3c(-c4c([He])c(C#N)cc5cccnc45)c4ccccc4cc3cc12'},
# 					{'smiles': '[He]c1c2cc3ccccc3c(-c3cc(C#N)cc4cccnc34)c2c([He])c2cc3ccccc3c(-c3cc(C#N)cc4cccnc34)c12'},
# 					{'smiles': '[He]c1cccc2cc3cc4c(-c5cc(C#N)cc6cccnc56)c5c([He])cccc5cc4cc3c(-c3cc(C#N)cc4cccnc34)c12'},
# 					{'smiles': '[He]c1ccnc2c(-c3c4ccccc4cc4cc5c(-c6cc(C#N)cc7c([He])ccnc67)c6ccccc6cc5cc34)cc(C#N)cc12'},
# 					{'smiles': '[He]c1ccc2cc3cc4c(-c5cc(C#N)cc6cccnc56)c5cc([He])ccc5cc4cc3c(-c3cc(C#N)cc4cccnc34)c2c1'},
# 					{'smiles': '[He]c1c(C#N)cc(-c2c3ccccc3cc3cc4c(-c5cc(C#N)c([He])c6cccnc56)c5ccccc5cc4cc23)c2ncccc12'},
# 					{'smiles': '[He]c1c2ccccc2c(-c2cc(C#N)cc3cccnc23)c2cc3c([He])c4ccccc4c(-c4cc(C#N)cc5cccnc45)c3cc12'}
# 				]
# 	}
# ]
