from rdkit import Chem
from rdkit.Chem import AllChem, Draw

# SMILES exemplu
smi = "CCO"  # etanol
mol = Chem.MolFromSmiles(smi)

bit_info = {}
fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024, bitInfo=bit_info)

if 33 in bit_info:
    print(f"Bit 33 activat de următoarea substructură:")
    for atom_idx, radius in bit_info[33]:
        env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
        submol = Chem.PathToSubmol(mol, env)
        img = Draw.MolToImage(submol)
        img.show()  # Afișează fereastra cu substructura
else:
    print("Bit 33 nu este prezent în această moleculă.")
