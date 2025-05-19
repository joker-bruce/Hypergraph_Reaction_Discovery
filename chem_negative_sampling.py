import numpy as np
import os
import random

from rdkit import Chem
from rdkit.Chem import rdmolfiles, rdmolops


def gen_neg_smiles(rxn_smiles):
    try:
        reactants = Chem.MolFromSmiles(rxn_smiles.split('>')[0])
        products  = Chem.MolFromSmiles(rxn_smiles.split('>')[2])
        conserved_maps = [
            a.GetProp('molAtomMapNumber')
            for a in products.GetAtoms() if a.HasProp('molAtomMapNumber')]
        bond_changes = set() 
        bonds_prev = {}
        bonds_prev_corre = {}## corresponding atom to its molatommapnumber
        #bonds_prev_corre_rev = {}## corresponding molatommapnumber to its atom
        for bond in reactants.GetBonds():
            nums = sorted(
                [bond.GetBeginAtom().GetProp('molAtomMapNumber'),
                bond.GetEndAtom().GetProp('molAtomMapNumber')])
            bonds_prev_corre[bond.GetBeginAtom().GetProp('molAtomMapNumber')] = bond.GetBeginAtom()
            bonds_prev_corre[bond.GetEndAtom().GetProp('molAtomMapNumber')] = bond.GetEndAtom()

            if (nums[0] not in conserved_maps) and (nums[1] not in conserved_maps):
                continue
            bonds_prev['{}~{}'.format(nums[0], nums[1])] = bond.GetBondTypeAsDouble()
        bonds_new = {}
        for bond in products.GetBonds():
            nums = sorted(
                [bond.GetBeginAtom().GetProp('molAtomMapNumber'),
                bond.GetEndAtom().GetProp('molAtomMapNumber')])
            bonds_new['{}~{}'.format(nums[0], nums[1])] = bond.GetBondTypeAsDouble()
        for bond in bonds_prev:
            if bond not in bonds_new:
                # lost bond
                bond_changes.add((bond.split('~')[0], bond.split('~')[1], 0.0))
            else:
                if bonds_prev[bond] != bonds_new[bond]:
                    # changed bond
                    bond_changes.add((bond.split('~')[0], bond.split('~')[1], bonds_new[bond]))
        for bond in bonds_new:
            if bond not in bonds_prev:
                # new bond
                bond_changes.add((bond.split('~')[0], bond.split('~')[1], bonds_new[bond]))

        #find bond containing the reaction center atoms
        center_atoms = set()
        for bond in bond_changes:
            center_atoms.add(bonds_prev_corre[bond[0]].GetIdx())
            center_atoms.add(bonds_prev_corre[bond[1]].GetIdx())
        bonds_contains_center_atoms = set()
        #print(center_atoms)
        for bond in reactants.GetBonds():
            #print(bond.GetBeginAtom())
            if bond.GetBeginAtom().GetIdx() in center_atoms or bond.GetEndAtom().GetIdx() in center_atoms:
                bonds_contains_center_atoms.add(bond)
        #print(f'bonds_contains_center_atoms{bonds_contains_center_atoms}')
        #generate negative samples
        neg_mol = bonds_modification(reactants, bonds_contains_center_atoms)###TODO: modify the aromatic bond to be single bond for all.
        ##check the validity of the negative samples
        validity = check_mol_validity(neg_mol)
        if validity == False:
            neg_mol = None

    except Exception as e:
        print(e)
        neg_mol = None
    if neg_mol is None:
        return None
    else:
        return Chem.MolToSmiles(neg_mol, canonical=True) + '>>'

def check_mol_validity(mol):
    #check the validity of the smiles string through iterating through reactants
    smiles = Chem.MolToSmiles(mol, canonical=True)
    try:
        reactants = smiles.split('.')
        for i in range(len(reactants)):
            mol = Chem.MolFromSmiles(reactants[i])
            if mol is None:
                return False
        return True
    except Exception as e:
        return False
def bonds_modification(reactants, bond_set):
    #print('len bond_set:',len(bond_set) )
    bond_to_modify = random.sample(bond_set, k=random.randint(1, len(bond_set)))
    modified_mol = reactants
    iter = 0
    break_outer_loop = False
    while iter < 50 and not break_outer_loop:  
        for bond in bond_to_modify:
            vir_mol = single_bond_modification(modified_mol, bond)
            if vir_mol is not None:
                modified_mol = vir_mol
            validity = check_mol_validity(modified_mol)
            if validity == True:
                break_outer_loop = True
                break
        iter += 1
    if modified_mol == reactants:
        return None
    else:
        return modified_mol
    
def single_bond_modification(reactants, bond, attempt=10):
    for i in range(attempt):
        rand = random.random()
        if rand < 0.25:
            ##remove bond here
            vir_mol =  Chem.RWMol(reactants)
            vir_mol.RemoveBond(bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx())
            if Chem.SanitizeMol(reactants, catchErrors=True) == Chem.SanitizeFlags.SANITIZE_NONE:
                return vir_mol
        else:
            bond_type = {Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC}
            vir_mol =  Chem.RWMol(reactants)
            ##choose a bond type to modify, excluding the original bond type
            if bond.GetBondType() == Chem.BondType.AROMATIC:##eliminate all of the aromatic bond to single bond
                for bond in vir_mol.GetBonds():
                    if bond.GetIsAromatic():
                        bond.SetBondType(Chem.BondType.SINGLE)
                        bond.SetIsAromatic(False) 
            else:
                bond_type.remove(bond.GetBondType())
                bond.SetBondType(random.choice(list(bond_type)))
            if Chem.SanitizeMol(reactants, catchErrors=True) == Chem.SanitizeFlags.SANITIZE_NONE:
                return vir_mol
    return None
