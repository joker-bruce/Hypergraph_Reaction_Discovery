import pandas as pd
import os
import json
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import multiprocessing
import re
import numpy as np
from rdkit import Chem
import torch
import negative_sampler as ns
from collections import defaultdict
import math
from tqdm.autonotebook import tqdm
import random
import joblib
from rdkit.Chem import AllChem
from rdkit.Chem import rdChemReactions
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
import util
import time
from chem_negative_sampling import gen_neg_smiles
import pdb

df = pd.read_csv("./train.txt", header=None, names=["smiles"])

##Hypergraph without reaction node
#global count
def process_row(row):
    #global count
#    global nodes
    count = 0
    count += 1
    #nodes_temp = pd.DataFrame(columns=['name', 'smiles', 'type', 'index'])
    nodes_temp = []
    try:
        pattern = re.compile(r'(?<![a-zA-Z])\'|\'(?![a-zA-Z])')
        #data = pattern.sub('"',row['molecules_reactants'])
        #data1= pattern.sub('"',row['molecules_products'])
        data2 = row['smiles']
        data = data2.split('>')[0].split('.')
        #print(data1)
        #data = json.loads(data)
        #data1 = json.loads(data1)
        #print(data)
        #data2 = row['smiles']
        #index = row['index']
        nodes_temp = []#pd.DataFrame(columns=columns)
        num_mol = 0
    except Exception as e:
        print(count, e)
        return nodes_temp
    new_data = []

    # for i,item in enumerate(data1):
    #     try:
    #         num_mol += 1
    #     except Exception as e:
    #         #print(f"Error processing row: {e}")
    #         num_mol += 1
    for i,item in enumerate(data):
        try:
            smiles = item
            # Parse the SMILES string
            mol = Chem.MolFromSmiles(smiles)

            # Generate the canonical SMILES
            canonical_smiles = Chem.MolToSmiles(mol)
            nodes_temp += [canonical_smiles]#pd.concat([nodes_temp,new_data])
        except Exception as e:
            continue
    return nodes_temp

def process_row_wrapper(chunk):
    #global lock
#    global nodes
    return chunk.progress_apply(lambda row: process_row(row), axis=1)

def remove_atom_mapping(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)  # Remove atom map number
    return Chem.MolToSmiles(mol)

def prep_data(datapoints, data_path, df, ns_algo, ns_ratio, output_dir, output_filename):
    df_hyper = df[:datapoints]#pd.read_csv(data_path)[:datapoints]
    df_hyper = df_hyper[['smiles']]
    neg_smiles_list = []
    for i in range(len(df_hyper)):
        for j in range(math.floor(ns_ratio)):
            smiles = df_hyper.at[i, 'smiles']
            #pdb.set_trace()
            neg_smiles = gen_neg_smiles(smiles)
            if neg_smiles is not None:
                neg_smiles_list.append(neg_smiles)
    neg_smiles_list_len = len(neg_smiles_list)
    neg_df = pd.DataFrame(neg_smiles_list, columns=['smiles'])
    neg_df.to_csv(f'neg_smiles_{ns_ratio}_{datapoints}.csv')
    df_hyper = pd.concat([df_hyper, neg_df], ignore_index=True)
    print('df_hyper len:', len(df_hyper))
    print('neg smiles list len:', neg_smiles_list_len)
    ##create a dir if there is none
    directory = output_dir
    output_filename = f'{output_filename}_{datapoints}'
# Check if the directory exists
    if not os.path.exists(directory):
    # Create the directory
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    results = []
    for i in tqdm(range(len(df_hyper))):
        data = df_hyper.at[i, 'smiles'].split('>')[0].split('.')
        new_data = []
        for j in data:
            #print(j)
            new_data.append(remove_atom_mapping(j))
        results.append(new_data)
    
    
    



    # chunk_size = (datapoints+neg_smiles_list_len)//10
    # print('chunk_size:', chunk_size)
    # tqdm.pandas(desc="Processing Rows")
    # #lock = multiprocessing.Lock()
    # #count = 0
    # for j in range(0,1):
    #     df_temp = df_hyper[j*chunk_size*10:]#(j+1)*chunk_size*10]
    #     chunks = [df_temp.iloc[i:min(i+chunk_size, len(df_temp))] for i in range(0, len(df_temp), chunk_size)]
    #     print('chunks len:', len(chunks))
    #     with ProcessPoolExecutor() as executor:
    #         results = list(tqdm(executor.map(process_row_wrapper,chunks), total=len(chunks), desc="Processing Chunks"))
    #print('results:', results)
    #return results
    data_no_feat, \
    data_ecfp4_feat, \
    data_ecfp6_feat, \
    mns_neg_hyperedges, \
    sns_neg_hyperedges, \
    cns_neg_hyperedges, \
    hyperedge_set, \
    neg_hyperedge_set = process_result(results, output_dir, output_filename, neg_smiles_list_len)   
    torch.save(data_no_feat, os.path.join(output_dir, f'{output_filename}_no_feat.pt'))
    torch.save(data_ecfp4_feat, os.path.join(output_dir, f'{output_filename}_ecfp4_feat.pt'))
    torch.save(data_ecfp6_feat, os.path.join(output_dir, f'{output_filename}_ecfp6_feat.pt'))
    return  data_no_feat,\
     data_ecfp4_feat,\
      data_ecfp6_feat,\
       mns_neg_hyperedges,\
        sns_neg_hyperedges,\
         cns_neg_hyperedges,\
          hyperedge_set, neg_hyperedge_set ##return the processed dict

def process_result(results, output_dir, output_filename, neg_smiles_list_len):
    hyperedge = results
    # for i in range(len(results)):
    #     print(len(results[i]))
    #     for j in range(len(results[i])):
    #         print('i, j:',i,j)
    #         hyperedge.append(results[i][j+i*len(results[i])])


    moldict = {}
    moldict_tf = {}
    count = 0
    for i, sublist in enumerate(hyperedge):
        for j, item in enumerate(sublist):
            if item in moldict.keys():
                continue
            else:
                if i < (len(hyperedge) - neg_smiles_list_len):
                    moldict[item] = count
                    count+=1
                    moldict_tf[item] = True
                else:
                    moldict[item] = count
                    count+=1
                    moldict_tf[item] = False

    node_smiles_list = list(moldict.keys())
    node_smiles_list_tf = list(moldict_tf.values())
    ##prepare dataset for generating negative edges
    hyperedge_set = transform_list_to_set(moldict, hyperedge)
    #print("content hyperedge_set:", hyperedge_set)
    #print('1st len(hyperedge_set)',len(list(hyperedge_set)))
    hyperedge_set, neg_hyperedge_set = list(hyperedge_set)[:-neg_smiles_list_len], list(hyperedge_set)[-neg_smiles_list_len:]
    #print('2nd len(hyperedge_set)',len(hyperedge_set))
    hyperedge_set, neg_hyperedge_set = set(hyperedge_set), set(neg_hyperedge_set)
    #print("2d content hyperedge_set:", hyperedge_set)
    #print('len(hyperedge_set)',len(hyperedge_set))
    #print('len(neg_hyperedge_set)',len(neg_hyperedge_set))
    mns_neg_hyperedges = ns.generate_negative_samples_for_hyperedges(
        hyperedge_set, 'MNS', 1 * len(hyperedge_set))
    #print('mns_neg_hyperedges',mns_neg_hyperedges)
   # print('len(hyperedge_set)',len(hyperedge_set))
   # print('hyperedge_set',hyperedge_set)
    sns_neg_hyperedges = ns.generate_negative_samples_for_hyperedges(
        hyperedge_set, 'SNS', 1 * len(hyperedge_set))
    cns_neg_hyperedges = ns.generate_negative_samples_for_hyperedges(
        hyperedge_set, 'CNS', 1 * len(hyperedge_set))
    # mns_neg_hyperedges = []
    # sns_neg_hyperedges = []
    # cns_neg_hyperedges = []

    prepare_training(list(hyperedge_set), list(neg_hyperedge_set), mns_neg_hyperedges, sns_neg_hyperedges, cns_neg_hyperedges, output_dir, output_filename)
    #print(hyperedge_set)

    node_edge_pair = []
    edge_node_pair = []
    n_edge = len(hyperedge_set) + len(neg_hyperedge_set)
    n_node = len(moldict)
    nodewt = torch.zeros(n_node,)
    edgewt = torch.zeros(n_edge,)
    ##get features
    node_feat = torch.zeros(n_node, 1024).numpy()
    ecfp4_feat = ECFP_feat(list(moldict.keys()), 2, 1024)
    ecfp6_feat = ECFP_feat(list(moldict.keys()), 3, 1024)

    for i, sublist in enumerate(list(hyperedge_set) + list(neg_hyperedge_set)):
        for j, item in enumerate(sublist):
            node_edge_pair.append([item,i])
            edge_node_pair.append([i,item])
            edgewt[i] += 1
            nodewt[item] += 1
    data_no_feat = {'node_smiles_list':node_smiles_list,'node_smiles_list_tf':node_smiles_list_tf, 'N_edges':n_edge, 'N_nodes':n_node, "NodeEdgePair":node_edge_pair, "EdgeNodePair":edge_node_pair, \
       'nodewt':nodewt, 'edgewt':edgewt, 'node_feat':node_feat}
    data_ecfp4_feat = {'node_smiles_list':node_smiles_list, 'node_smiles_list_tf':node_smiles_list_tf, 'N_edges':n_edge, 'N_nodes':n_node, "NodeEdgePair":node_edge_pair, "EdgeNodePair":edge_node_pair, \
       'nodewt':nodewt, 'edgewt':edgewt, 'node_feat':ecfp4_feat}
    data_ecfp6_feat = {'node_smiles_list':node_smiles_list, 'node_smiles_list_tf':node_smiles_list_tf, 'N_edges':n_edge, 'N_nodes':n_node, "NodeEdgePair":node_edge_pair, "EdgeNodePair":edge_node_pair, \
       'nodewt':nodewt, 'edgewt':edgewt, 'node_feat':ecfp6_feat}




    return data_no_feat, data_ecfp4_feat, data_ecfp6_feat, mns_neg_hyperedges, sns_neg_hyperedges, cns_neg_hyperedges, hyperedge_set, neg_hyperedge_set

def transform_list_to_set(moldict, hyperedge):
    hyperedge_set = set()
    for i, sublist in enumerate(hyperedge):
        temp_list = []
        if len(sublist) == 0:
            continue
        for j, item in enumerate(sublist):
            temp_list.append(moldict[item])
        hyperedge_set.add(frozenset(temp_list))
        del temp_list
    return hyperedge_set



### ECFP4

def ECFP_feat(smiles_list, radius, feature_size):
    ## if radius is 2, then ECFP4, if 3, then ECFP6
    feature_matrix = []
    for i, item in enumerate(smiles_list):
        reactant = item
        dest_array = np.zeros((1,))
        ## convert the bit vector into numpy array and store in dest_array
        ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(reactant), radius, feature_size),dest_array)
        feature_matrix.append(dest_array.tolist())
    return torch.tensor(feature_matrix)
def get_union(union):
    ind = []
    for s in union :
        ind+=list(s)
    #print(set(ind))
    return set(ind)

def set_cover(universe, subsets):
    elements = set(e for s in subsets for e in s)
    if elements != universe:
        return None, None
    covered = set()
    cover = []
    idx = []
    while covered != elements:
        subset = max(subsets, key=lambda s: len(s - covered))
        cover.append(subset)
        idx.append(subsets.index(subset))
        covered |= subset
    return cover, idx

def get_cover_idx(HE):
    universe = get_union(HE)
    #print('universe',len(universe))
    tmp_HE = [set(edge) for edge in HE]
    _, cover_idx = set_cover(universe, tmp_HE)
    #print('cover_idx', len(cover_idx))
    #print(cover_idx)
    return cover_idx


def prepare_training(HE_set, HE_neg_set, mns_neg, sns_neg, cns_neg, output_dir, output_filename):
    HE = HE_set
    # base_cover = get_cover_idx(HE)##which edges can cover all of the elements
    # union = get_union(HE)##used for cheking if elements are same
    # tmp = [HE[idx] for idx in base_cover]
    # assert union == get_union(tmp)
    # base_num = len(base_cover)
    HE_neg = HE_neg_set
    for split in range(5):
        #print('base_num', base_num)
        # if base_num < int(0.6*len(HE)):## if the base number is smaller than the training set
        #     ##we will take the 0.6 of all hyperedge
        ground_num = int(0.6*len(HE))## remove the edge with coverage
        ground_neg_num = int(0.6*len(HE_neg))
        # else:## if the base number is more than the traning set
        #     ground_num = base_num

        total_idx = list(range(len(HE))) ##length of HE, total number of edge
        total_neg_idx = list(range(len(HE_neg))) ##length of HE_neg, total number of negative edge

        # if ground_num != base_num:## sample only when the base number is smaller than the training set
        #     ground_idx = list(set(total_idx)-set(base_cover))
        ground_idx = random.sample(total_idx, ground_num)
        ground_neg_idx = random.sample(total_neg_idx, ground_neg_num)
        # else:
        #     ground_idx = list(set(base_cover))

        print('ground_idx length',len(ground_idx), 'ground_num', ground_num)
        print('ground_neg_idx length',len(ground_neg_idx), 'ground_neg_num', ground_neg_num)


        # if ground_num != base_num:
        #     ground_num += base_num
        #     ground_idx += base_cover

        ground_valid_num = ground_num//6
        ground_neg_valid_num = ground_neg_num//6
        ground_valid_idx = random.sample(ground_idx, ground_valid_num)
        ground_neg_valid_idx = random.sample(ground_neg_idx, ground_neg_valid_num)


        ground_train_num = ground_num - ground_valid_num
        ground_neg_train_num = ground_neg_num - ground_neg_valid_num

        ground_train_data = []
        ground_valid_data = []
        pred_data = []
        ground_neg_train_data = []
        ground_neg_valid_data = []
        pred_neg_data = []
        for idx in total_idx :
            if idx in ground_idx:
                if idx in ground_valid_idx:
                    ground_valid_data.append(list(HE[idx]))
                else:
                    ground_train_data.append(list(HE[idx]))
            else :
                pred_data.append(list(HE[idx]))

        for idx in total_neg_idx :
            if idx in ground_neg_idx:
                if idx in ground_neg_valid_idx:
                    ground_neg_valid_data.append(list(HE_neg[idx]))
                else:
                    ground_neg_train_data.append(list(HE_neg[idx]))
            else :
                pred_neg_data.append(list(HE_neg[idx]))

        valid_only_num = int(0.25*len(pred_data))##10%
        train_only_num = int(0.25*len(pred_data))##10%
        test_num = len(pred_data) - (valid_only_num + train_only_num)
        valid_neg_only_num = int(0.25*len(pred_neg_data))##10%
        train_neg_only_num = int(0.25*len(pred_neg_data))##10%
        test_neg_num = len(pred_neg_data) - (valid_neg_only_num + train_neg_only_num)

        random.shuffle(pred_data)
        train_only_data = pred_data[:train_only_num]
        valid_only_data = pred_data[train_only_num:-test_num]
        test_data = pred_data[-test_num:]
        random.shuffle(pred_neg_data)
        train_neg_only_data = pred_neg_data[:train_neg_only_num]
        valid_neg_only_data = pred_neg_data[train_neg_only_num:-test_neg_num]
        test_neg_data = pred_neg_data[-test_neg_num:]

        train_mns, valid_mns, test_mns = read_neg_data(mns_neg)#, 'mns')
        train_sns, valid_sns, test_sns = read_neg_data(sns_neg)#, 'sns')
        train_cns, valid_cns, test_cns = read_neg_data(cns_neg)#, 'cns')

        print(f"ground {len(ground_train_data)} + {len(ground_valid_data)} = {len(ground_train_data + ground_valid_data)}")
        print(f"train pos {len(ground_train_data)} + {len(train_only_data)} = {len(ground_train_data + train_only_data)}, neg {len(train_sns)}")
        print(f"valid pos {len(ground_valid_data)} + {len(valid_only_data)} = {len(ground_valid_data + valid_only_data)}, neg {len(valid_sns)}")
        print(f"test pos {len(test_data)}, neg {len(test_sns)}")

        print(f"ground neg {len(ground_neg_train_data)} + {len(ground_neg_valid_data)} = {len(ground_neg_train_data + ground_neg_valid_data)}")
        print(f"train neg {len(ground_neg_train_data)} + {len(train_neg_only_data)} = {len(ground_neg_train_data + train_neg_only_data)}")
        print(f"valid neg {len(ground_neg_valid_data)} + {len(valid_neg_only_data)} = {len(ground_neg_valid_data + valid_neg_only_data)}")
        print(f"test neg {len(test_neg_data)}")

        # neg_dic = {'ground_train': ground_train_data, 'ground_valid': ground_valid_data, \
        #         'train_only_pos': train_only_data, 'train_mns': train_mns, 'train_sns' : train_sns, 'train_cns' : train_cns,\
        #         'valid_only_pos': valid_only_data, 'valid_mns': valid_mns, 'valid_sns' : valid_sns, 'valid_cns' : valid_cns, \
        #         'test_pos': test_data, 'test_mns': test_mns, 'test_sns' : test_sns, 'test_cns' : test_cns}
        # return neg_dic
        split_directory = os.path.join(output_dir, 'splits')
        if not os.path.exists(split_directory):
    # Create the directory
            os.makedirs(split_directory)
            print(f"Directory '{split_directory}' created.")

        full_path = os.path.join(output_dir, 'splits', output_filename)
        torch.save({'ground_train': ground_train_data, 'ground_valid': ground_valid_data, \
                'ground_neg_train': ground_neg_train_data, 'ground_neg_valid': ground_neg_valid_data, \
                'train_only_pos': train_only_data, 'train_only_neg': train_neg_only_data, 'train_mns': train_mns, 'train_sns' : train_sns, 'train_cns' : train_cns,\
                'valid_only_pos': valid_only_data, 'valid_only_neg': valid_neg_only_data,'valid_mns': valid_mns, 'valid_sns' : valid_sns, 'valid_cns' : valid_cns, \
                'test_pos': test_data, 'test_neg': test_neg_data, 'test_mns': test_mns, 'test_sns' : test_sns, 'test_cns' : test_cns},
                f'{full_path}split{split}.pt')
        

def read_csv_to_list(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        #data = [row for row in reader]
        data = [[int(item) for item in row] for row in reader]
    return data


def read_neg_data(neg_hyperedge):#, neg_type):
    #data = read_csv_to_list(f'{neg_dir}{neg_type}_hyperedge_neg.csv')
    if neg_hyperedge == []:
        return [], [], []
    data = neg_hyperedge
#     sns_data = read_csv_to_list(f'sns{neg_path}')
#     cns_data = read_csv_to_list(f'cns{neg_path}')
    neg_num, total_idx= int(0.6*len(data)), list(range(len(data)))
#     sns_neg_num, sns_total_idx= int(0.6*len(sns_data)), list(range(len(sns_data)))
#     cns_neg_num, cns_total_idx= int(0.6*len(cns_data)), list(range(len(cns_data)))
    neg_idx = random.sample(total_idx, neg_num)
    neg_valid_num = neg_num//6
    neg_valid_idx = random.sample(neg_idx, neg_valid_num)
    neg_train_num = neg_num - neg_valid_num
#     sns_neg_idx = random_sample(sns_total_idx, sns_neg_num)
#     cns_neg_idx = random_sample(cns_total_idx, cns_neg_num)
    neg_train_data = []
    neg_valid_data = []
    pred_data = []
    for idx in total_idx:
        if idx in neg_idx:
            if idx in neg_valid_idx:
                neg_valid_data.append(data[idx])
            else:
                neg_train_data.append(data[idx])
        else :
            pred_data.append(data[idx])

    valid_only_num = int(0.25*len(pred_data))##10%
    train_only_num = int(0.25*len(pred_data))##10%
    test_num = len(pred_data) - (valid_only_num + train_only_num)

    random.shuffle(pred_data)
    train_only_data = pred_data[:train_only_num]
    valid_only_data = pred_data[train_only_num:-test_num]
    test_data = pred_data[-test_num:]
    neg_train_data += train_only_data
    neg_valid_data += valid_only_data
    return neg_train_data, neg_valid_data, test_data






# Start the timer

if __name__ == "__main__":
    args = util.parse_args()
    start_time = time.time()
    data_no_feat, data_ecfp4_feat, data_ecfp6_feat, mns_neg_hyperedges, sns_neg_hyperedges, cns_neg_hyperedges, hyperedge_set, neg_hyperedge_set = prep_data(args.datapoints, args.datapath, df, 'mns', args.ns_ratio, args.output_dir, args.output_filename)
    end_time = time.time()
    (end_time - start_time)/60
