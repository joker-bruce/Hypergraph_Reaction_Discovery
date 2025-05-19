# Install required packages.
import os
import torch
from sklearn import metrics
#import pdb
os.environ['TORCH'] = torch.__version__
os.environ['DGLBACKEND'] = "pytorch"

# Uncomment below to install required packages. If the CUDA version is not 11.8,
# check the https://www.dgl.ai/pages/start.html to find the supported CUDA
# version and corresponding command to install DGL.
#!pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html > /dev/null
#!pip install torchmetrics > /dev/null

try:
    import dgl
    installed = True
except ImportError:
    installed = False
print("DGL installed!" if installed else "Failed to install DGL!")

import dgl.sparse as dglsp
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
#from dgl.data import CoraGraphDataset
from torchmetrics.functional import accuracy
import numpy as np
from torchmetrics import Accuracy, Precision, Recall, F1Score, Specificity
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from torchmetrics import AveragePrecision
from util import smiles_to_pretrained_gnn_data, reaction_center_config, reshape_tensor
from dgllife.model import load_pretrained, WLNReactionCenter
import wandb
import argparse
import copy 
class AttentionAggregator(nn.Module):
    def __init__(self, dim_vertex,dim_gnn_embedding, dim_key_val, layers, aggre_method):
        super(AttentionAggregator, self).__init__()
        self.d_k = self.d_v = dim_key_val  # Dimension of keys/values
        self.n = dim_vertex # embedding size per node
        self.W_Q = nn.Linear(self.n, self.d_k)
        self.W_K = nn.Linear(dim_gnn_embedding, self.d_k)
        self.W_V = nn.Linear(dim_gnn_embedding, self.d_v)
        self.aggre_method = aggre_method
        layers = copy.deepcopy(layers)
        layers[0] += self.d_v
        Layers = []
        for i in range(len(layers)-1):
            Layers.append(nn.Linear(layers[i], layers[i+1]))
            if i != len(layers)-2:
                Layers.append(nn.ReLU(True))
        self.cls = nn.Sequential(*Layers)
    def classify(self, embedding):
        return F.sigmoid(self.cls(embedding))

    def forward(self, gnn_embedding, hgnn_embedding):
        #gnn embedding is the output from the GNN of m*m*5
        #hgnn output is the output from hgnn and stacked into 2d matrix k*n 
        #where n is the embeeding of nodes and the k is the number of nodes involved in the reaction
        #however, they are projected into a larger 
        #self.cls = nn.Sequential(*Layers)
        # print(f"gnn_embedding shape:{gnn_embedding.shape}\n\
        #     hgnn_embedding shape:{hgnn_embedding.shape}")
        Q = self.W_Q(hgnn_embedding)
        K = self.W_K(gnn_embedding)
        V = self.W_V(gnn_embedding)
        attention_scores = torch.matmul(Q, K.T) / (self.d_k ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        # Compute merged representation
        merged = torch.matmul(attention_weights, V)  # [k, d_v]

        # Optionally combine with the original Matrix 1
        embeddings = torch.cat([hgnn_embedding, merged], dim=-1)  # [k, n + d_v]
        if self.aggre_method == 'mean':
            final_embedding = torch.mean(embeddings, dim=0)
        elif self.aggre_method == 'maxmin':
            max_val, _ = torch.max(embeddings, dim=0)
            min_val, _ = torch.min(embeddings, dim=0)
            final_embedding = max_val - min_val
        elif self.aggre_method == 'max':
            final_embedding, _ = torch.max(embeddings, dim=0)
        elif self.aggre_method == 'min':
            final_embedding, _ = torch.min(embeddings, dim=0)
        elif self.aggre_method == 'sum':
            final_embedding = torch.sum(embeddings, dim=0)
        else:
            assert False, "No such aggregator method, please choose from mean, maxmin, max, min, sum"
        pred = self.classify(final_embedding)
        return pred, final_embedding
        
class MaxminAggregator(nn.Module):
    def __init__(self, dim_vertex, layers):
        super(MaxminAggregator, self).__init__()
        Layers = []
        for i in range(len(layers)-1):
            Layers.append(nn.Linear(layers[i], layers[i+1]))
            if i != len(layers)-2:
                Layers.append(nn.ReLU(True))
        self.cls = nn.Sequential(*Layers)
    
    def aggregate(self, embeddings):
        max_val, _ = torch.max(embeddings, dim=0)
        min_val, _ = torch.min(embeddings, dim=0)
        return max_val - min_val
    
    def classify(self, embedding):
        return F.sigmoid(self.cls(embedding))
    
    def forward(self, embeddings):
        embedding = self.aggregate(embeddings)
        pred = self.classify(embedding)
        return pred, embedding


class HEBatchGenerator(object):
    def __init__(self, hyperedges, labels, batch_size, device, test_generator=False):
        """Creates an instance of HyperedgeGroupBatchGenerator.
        
        Args:
            hyperedges: List(frozenset). List of hyperedges.
            labels: list. Labels of hyperedges.
            batch_size. int. Batch size of each batch.
            test_generator: bool. Whether batch generator is test generator.
        """
        self.batch_size = batch_size
        self.hyperedges = hyperedges
        self.labels = labels
        self._cursor = 0
        self.device = device
        self.test_generator = test_generator
        self.shuffle()
    
    def eval(self):
        self.test_generator = True

    def train(self):
        self.test_generator = False

    def shuffle(self):
        idcs = np.arange(len(self.hyperedges))
        np.random.shuffle(idcs)
        self.hyperedges = [self.hyperedges[i] for i in idcs]
        self.labels = [self.labels[i] for i in idcs]
  
    def __iter__(self):
        self._cursor = 0
        return self
    
    def next(self):
        return self.__next__()

    def __next__(self):
        if self.test_generator:
            return self.next_test_batch()
        else:
            return self.next_train_batch()

    def next_train_batch(self):
        ncursor = self._cursor+self.batch_size
        if ncursor >= len(self.hyperedges):
            hyperedges = self.hyperedges[self._cursor:] + self.hyperedges[
                :ncursor - len(self.hyperedges)]

            labels = self.labels[self._cursor:] + self.labels[
                :ncursor - len(self.labels)]
          
            self._cursor = ncursor - len(self.hyperedges)
            hyperedges = [torch.LongTensor(list(edge)).to(self.device) for edge in hyperedges]
            labels = torch.FloatTensor(labels).to(self.device)
            self.shuffle()
            return hyperedges, labels, True
        
        hyperedges = self.hyperedges[
            self._cursor:self._cursor + self.batch_size]
        
        labels = self.labels[
            self._cursor:self._cursor + self.batch_size]
        
        hyperedges = [torch.LongTensor(list(edge)).to(self.device) for edge in hyperedges]
        labels = torch.FloatTensor(labels).to(self.device)
       
        self._cursor = ncursor % len(self.hyperedges)
        return hyperedges, labels, False

    def next_test_batch(self):
        ncursor = self._cursor+self.batch_size
        if ncursor >= len(self.hyperedges):
            hyperedges = self.hyperedges[self._cursor:]
            labels = self.labels[self._cursor:]
            self._cursor = 0
            hyperedges = [torch.LongTensor(list(edge)).to(self.device) for edge in hyperedges]
            labels = torch.FloatTensor(labels).to(self.device)
            
            return hyperedges, labels, True
        
        hyperedges = self.hyperedges[
            self._cursor:self._cursor + self.batch_size]
        
        labels = self.labels[
            self._cursor:self._cursor + self.batch_size]

        hyperedges = [torch.LongTensor(list(edge)).to(self.device) for edge in hyperedges]
        labels = torch.FloatTensor(labels).to(self.device)
       
        self._cursor = ncursor % len(self.hyperedges)
        return hyperedges, labels, False

class HGNN(nn.Module):
    def __init__(self, H, in_size, out_size, device, hidden_dims=16):
        super().__init__()

        self.W1 = nn.Linear(in_size, hidden_dims)
        self.W2 = nn.Linear(hidden_dims, out_size)
        self.dropout = nn.Dropout(0.5)

        ###########################################################
        # (HIGHLIGHT) Compute the Laplacian with Sparse Matrix API
        ###########################################################
        # Compute node degree.
        d_V = H.sum(1)
        # Compute edge degree.
        d_E = H.sum(0)
        # Compute the inverse of the square root of the diagonal D_v.
        D_v_invsqrt = torch.diag(d_V**-0.5)
        # Compute the inverse of the diagonal D_e.
        D_e_inv = torch.diag(d_E**-1.0)
        # In our example, B is an identity matrix.
        n_edges = d_E.shape[0]
        B = torch.eye(n_edges).to(device)
        #print(f"param shape:{d_V.shape},{d_E.shape},{D_v_invsqrt.shape},{D_e_inv.shape}")
        # Compute Laplacian from the equation above.
        self.L = D_v_invsqrt @ H @ B @ D_e_inv @ H.T @ D_v_invsqrt
        
        element_size = self.L.element_size()  # Number of bytes per element
        tensor_size = self.L.nelement()  # Number of elements in the tensor
        print(f"Size of tensor: {element_size * tensor_size} bytes")

    def forward(self, X):
        # print(f"self.L shape:{self.L.shape}\n\
        #     self.dropout{self.dropout(X).shape}")
        X = self.L @ self.W1(self.dropout(X))
        X = F.relu(X)
        X = self.L @ self.W2(self.dropout(X))
        return X
    
##code about the aggregator


class HGNN1(nn.Module):
    def __init__(self, H, in_size, out_size, device, hidden_dims=16):
        super().__init__()

        self.W1 = nn.Linear(in_size, hidden_dims)
        self.W2 = nn.Linear(hidden_dims, out_size)
        self.dropout = nn.Dropout(0.5)

        ###########################################################
        # (HIGHLIGHT) Compute the Laplacian with Sparse Matrix API
        ###########################################################
        # Compute node degree.
        d_V = H.sum(1)
        # Compute edge degree.
        d_E = H.sum(0)
        # Compute the inverse of the square root of the diagonal D_v.
        D_v_invsqrt = dglsp.diag(d_V**-0.5)
        # Compute the inverse of the diagonal D_e.
        D_e_inv = dglsp.diag(d_E**-1)
        # In our example, B is an identity matrix.
        n_edges = d_E.shape[0]
        indices = torch.arange(n_edges)
        values = torch.ones(n_edges)

        # Use COO format for the identity matrix
        identity_sparse = dglsp.from_coo(indices, indices, values)
        #B = identity_sparse.to(device)#dglsp.identity((n_edges, n_edges))
        # Compute Laplacian from the equation above.
        self.L = D_v_invsqrt @ H @ D_e_inv @ H.T @ D_v_invsqrt#D_v_invsqrt @ H @ B @ D_e_inv @ H.T @ D_v_invsqrt

    def forward(self, X):
        X = self.L @ self.W1(self.dropout(X))
        X = F.relu(X)
        X = self.L @ self.W2(self.dropout(X))
        return X

def load_data(data_path):
    #dataset = CoraGraphDataset()
    #data_dict = torch.load('data_path')
    data_dict = torch.load(data_path)
    #NodeEdgePair = torch.LongTensor(data_dict['NodeEdgePair'])
    EdgeNodePair = torch.LongTensor(data_dict['EdgeNodePair'])
    ne = data_dict['N_edges']
    nv = data_dict['N_nodes']
    node_feat = data_dict['node_feat']
    node_smiles_list = data_dict['node_smiles_list']
    #nodewt = data_dict['nodewt']
    #edgewt = data_dict['edgewt']
    
    #graph = dataset[0]
    #indices = torch.stack(graph.edges())
    H = torch.zeros(ne,nv)
    for i, item in enumerate(EdgeNodePair):
        H[item[0], item[1]] = 1
    # H = dglsp.spmatrix(indices)
    # H = H + dglsp.identity(H.shape)
    X = node_feat
    #X = graph.ndata["feat"]
    #Y = graph.ndata["label"]
    #train_mask = graph.ndata["train_mask"]
    #val_mask = graph.ndata["val_mask"]
    #test_mask = graph.ndata["test_mask"]
    return H, torch.tensor(X), node_smiles_list #dataset.num_classes, train_mask, val_mask, test_mask

def load_splits(split_path):
    split_dict = torch.load(split_path)
    return split_dict

def load_train(data_dict, bs, device, label):
    if label == 'chem_neg':
        train_pos = data_dict["train_only_pos"] + data_dict["ground_train"]
        #train_neg = data_dict["train_sns"]
        train_neg = data_dict['train_only_neg'] + data_dict["ground_neg_train"]
    elif label == 'mix':
        train_pos = data_dict["train_only_pos"] + data_dict["ground_train"]
        l = len(train_pos)//4
        train_neg = data_dict["train_sns"][0:l] + data_dict["train_mns"][:l] + data_dict["train_cns"][:l] + (data_dict['train_only_neg'] + data_dict["ground_neg_train"])[:l]
    elif label == 'mix_no_chem':
        train_pos = data_dict["train_only_pos"] + data_dict["ground_train"]
        l = len(train_pos)//3
        train_neg = data_dict["train_sns"][0:l] + data_dict["train_mns"][:l] + data_dict["train_cns"][:l]
    else:
        if label not in ['mns','sns', 'cns']:
            assert False, "No Such Train Label, please note the labels are one of \
the following: mns, sns, cns, chem_neg"
        train_pos = data_dict["train_only_pos"] + data_dict["ground_train"]
        train_neg = data_dict[f"train_{label}"]
        
    train_pos_label = [1 for i in range(len(train_pos))]
    train_neg_label = [0 for i in range(len(train_neg))]
    train_batchloader = HEBatchGenerator(train_pos + train_neg, train_pos_label + train_neg_label, bs, device, test_generator=False)    
    return train_batchloader

def load_val(data_dict, bs, device, label):
    if label=="pos":
        val = data_dict["valid_only_pos"] + data_dict["ground_valid"]
        val_label = [1 for i in range(len(val))]
    elif label=="chem_neg":
        val = data_dict["valid_only_neg"] + data_dict["ground_neg_valid"]
        val_label = [0 for i in range(len(val))]
    else:
        val = data_dict[f"valid_{label}"]
        val_label = [0 for i in range(len(val))]
    val_batchloader = HEBatchGenerator(val, val_label, bs, device, test_generator=True)    
    return val_batchloader

def load_test(data_dict, bs, device, label):
    
    if label=="pos":
        test = data_dict[f"test_pos"]
        test_label = [1 for i in range(len(test))]
    elif label=="chem_neg":
        test = data_dict[f"test_neg"] + data_dict[f"test_pos"]
        test_label = [0 for i in range(len(data_dict[f"test_neg"]))] + [1 for i in range(len(data_dict[f"test_pos"]))]
    else:
        test = data_dict[f"test_{label}"] + data_dict[f"test_pos"]
        test_label = [0 for i in range(len(data_dict[f"test_{label}"]))] + [1 for i in range(len(data_dict[f"test_pos"]))]
    test_batchloader = HEBatchGenerator(test, test_label, bs, device, test_generator=True)    
    return test_batchloader

def get_reactant_smiles(hedge, node_smiles_list):
    li = [node_smiles_list[node] for node in hedge]
    result = '.'.join(li)
    return result
def pretrained_gnn_embedding(mol_graph, complete_graph, device, pretrained_gnn_model):
    mol_graphs = mol_graph.to(device)
    complete_graphs = complete_graph.to(device)
    node_feats = mol_graphs.ndata.pop('hv').to(device)
    if mol_graphs.num_edges() > 0:
        edge_feats = mol_graphs.edata.pop('he').to(device)
    else:
        edge_feats = torch.zeros((0, pretrained_gnn_model.gnn.project_edge_messages.in_feats), device=device)
    node_pair_feats = complete_graphs.edata.pop('feats').to(device)
    return pretrained_gnn_model(mol_graphs, complete_graphs, node_feats, edge_feats, node_pair_feats)

def load_pretrained_gnn_model(model_path, device, args):
    checkpoint = torch.load(model_path, map_location=device)
    pretrained_gnn_model = WLNReactionCenter(node_in_feats=args['node_in_feats'],
                              edge_in_feats=args['edge_in_feats'],
                              node_pair_in_feats=args['node_pair_in_feats'],
                              node_out_feats=args['node_out_feats'],
                              n_layers=args['n_layers'],
                              n_tasks=args['n_tasks']).to(device)
    pretrained_gnn_model.load_state_dict(checkpoint['model_state_dict'])
    return pretrained_gnn_model
def train(args, model, pretrained_gnn_model, optimizer, X, node_smiles_list, train_hedges, train_labels, aggregator, device):
    model.train()
    pretrained_gnn_model.eval()
    X_hat = model(X)
    train_pred = []
    train_embed = []
    #print("train_hedge type:", type(train_hedges))
    for hedge in train_hedges:
        embedding = X_hat[hedge]
        reactant_smiles = get_reactant_smiles(hedge, node_smiles_list)
        mol_graph, complete_graph = smiles_to_pretrained_gnn_data(reactant_smiles)
        with torch.no_grad():
            _, biased_pred = pretrained_gnn_embedding(
                mol_graph, complete_graph, device, pretrained_gnn_model)
        pred, embed = aggregator(biased_pred,embedding)
        # del biased_pred
        # del _
        # embed = embed.detach().cpu()
        # #pred = pred.detach().cpu()
        # torch.cuda.empty_cache()
        # #del embed
        train_embed.append(embed)
        train_pred.append(pred)
    del X_hat
    train_pred = torch.stack(train_pred).squeeze().to(train_labels.device)
    train_embed = torch.stack(train_embed).to(device)
    embed_labels = torch.zeros(train_embed.shape[0], train_embed.shape[1]).to(device)
    # print(f"train_pred shape:{train_pred.shape}\n\
    #     train_labels shape:{train_labels.shape}")
    # print(f"train_pred:{train_pred}\n\
    #     train_labels:{train_labels}")
    lambda1, lambda2 = args.Lambda, 1-args.Lambda
    if args.loss_type == 'mse':
        loss = lambda1 * F.binary_cross_entropy(train_pred, train_labels) + lambda2 * F.mse_loss(train_embed, embed_labels, reduction='mean')
    elif args.loss_type == 'rmse':
        loss = lambda1 * F.binary_cross_entropy(train_pred, train_labels) + lambda2 * F.mse_loss(train_embed, embed_labels, reduction='mean').sqrt()
    else:
        assert False, "No such loss type, please choose from rmse or mse"
    loss = lambda1 * F.binary_cross_entropy(train_pred, train_labels) + lambda2 * F.mse_loss(train_embed, embed_labels)
    #print(f"loss:{loss}") 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss, train_pred, train_labels


def model_eval(model, pretrained_gnn_model, X, node_smiles_list, test_batchloader, aggregator, device):
    model.eval()
    pretrained_gnn_model.eval()
    X_hat = model(X)
    test_pred, test_label = [], []
    while True:
        hedges, labels, is_last = test_batchloader.next()
        for hedge, label in zip(hedges,labels):
            embedding = X_hat[hedge]
            reactant_smiles = get_reactant_smiles(hedge, node_smiles_list)
            mol_graph, complete_graph = smiles_to_pretrained_gnn_data(reactant_smiles)
            with torch.no_grad():
                _, biased_pred = pretrained_gnn_embedding(
                    mol_graph, complete_graph, device, pretrained_gnn_model)
            pred, embed = aggregator(biased_pred, embedding)
            test_pred.append(pred)
            test_label.append(label)
        if is_last:
            break
    test_pred, test_label = torch.stack(test_pred).squeeze(), torch.stack(test_label).squeeze()
    loss = F.binary_cross_entropy(test_pred, test_label)

    return test_pred,test_label,loss

def print_eval(test_pred,test_label,loss):
    accuracy_metric = Accuracy(task='binary')
    precision_metric = Precision(task='binary')
    recall_metric = Recall(task='binary')
    f1_metric = F1Score(task='binary')
    specificity_metric = Specificity(task='binary')
    accuracy = accuracy_metric(test_pred, test_label)
    precision = precision_metric(test_pred, test_label)
    recall = recall_metric(test_pred, test_label)
    f1 = f1_metric(test_pred, test_label)
    specificity = specificity_metric(test_pred, test_label)
    auc_roc = metrics.roc_auc_score(test_label.detach().numpy(), test_pred.detach().numpy())
    print(f'loss:{loss}\n \
    accuracy:{accuracy}\n \
    precision:{precision}\n \
    recall:{recall}\n \
    f1:{f1}\n \
    specificity:{specificity}\n \
    auc_roc:{auc_roc}\n')

    return f'loss:{loss}, accuracy:{accuracy}, precision:{precision}, recall:{recall}, f1:{f1}, specificity:{specificity}\n '
def get_eval(test_pred,test_label,loss):
    accuracy_metric = Accuracy(task='binary')
    precision_metric = Precision(task='binary')
    recall_metric = Recall(task='binary')
    f1_metric = F1Score(task='binary')
    specificity_metric = Specificity(task='binary')
    accuracy = accuracy_metric(test_pred, test_label)
    precision = precision_metric(test_pred, test_label)
    recall = recall_metric(test_pred, test_label)
    f1 = f1_metric(test_pred, test_label)
    specificity = specificity_metric(test_pred, test_label)
    auc_roc = metrics.roc_auc_score(test_label.detach().numpy(), test_pred.detach().numpy())
    # print(f'loss:{loss}\n \
    # accuracy:{accuracy}\n \
    # precision:{precision}\n \
    # recall:{recall}\n \
    # f1:{f1}\n \
    # specificity:{specificity}\n \
    # auc_roc:{auc_roc}\n')

    return loss, accuracy, precision, recall, f1, specificity, auc_roc

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='cora', help='dataset name')
    #parser.add_argument("--dataset_point", type=int, default=10000, help='# of datapoints')
    parser.add_argument('--datapath', type=str, default='./data_neg_balanced_smiles')
    parser.add_argument("--gpu", type=int, default=0, help='gpu number. -1 if cpu else gpu number')
    parser.add_argument("--exp_num", default=1, type=int, help='number of experiments')
    parser.add_argument("--epochs", default=50, type=int, help='number of epochs')
    parser.add_argument("--bs", default=16, type=int, help='batch size')
    parser.add_argument("--folder_name", type=str, default='exp', help='experiment name')
    parser.add_argument("--aggre_method", type=str, default='maxmin', help='aggregator method: mean, maxmin, max, min, sum')
    parser.add_argument("--encoding", type=str, default='no_feat', help='encoding type (no_feat, ecfp4_feat, ecfp6_feat)')
    parser.add_argument("--dim_vertex", default=1024, type=int, help='dimension of vertex hidden vector')
    parser.add_argument("--dim_gnn_embedding", default=5, type=int, help='dimension of gnn embedding')
    parser.add_argument("--train_label", type=str, default='mix', help='training label (mix, chem_neg, mns, sns, cns)')
    parser.add_argument("--dim_key_val", default=256, type=int, help='dimension of key and value')
    parser.add_argument("--lr", default=0.0001, type=float, help='learning rate')
    parser.add_argument("--Lambda", default=0.5, type=float, help='Lambda for weights of losses')
    parser.add_argument("--loss_type", default='rmse', type=str, help='loss type for embedding, rmse or mse')
    parser.add_argument("--cls_layers", default=[1024, 256, 16, 1], type=list, help='classifier layers')
    parser.add_argument("--split", type=int, default=0, help='split number: 0,1,2,3,4')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    args_pretrained = args.__dict__
    args_pretrained.update(reaction_center_config)
    #print(args)
    device = 'cuda:{}'.format(args.gpu) if args.gpu != -1 else 'cpu'
    data_path = f'{args.datapath}/{args.dataset_name}_{args.encoding}.pt'
    split_path = f'{args.datapath}/splits/{args.dataset_name}split{args.split}.pt'
    log_path = f"{args.datapath}/checkpoints/{args.folder_name}_split{args.split}/logs"
    os.makedirs(f"{args.datapath}/checkpoints/{args.folder_name}_split{args.split}", exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    #os.makedirs(f"./data/checkpoints/{args.folder_name}", exist_ok=False)
    f_log = open(f"{log_path}/{args.folder_name}_train.log", "w")
    f_log.write(f"args: {args}\n")

    H, X, node_smiles_list = load_data(data_path) #Y, num_classes, train_mask, val_mask, test_mask = load_data()

    coordinates = torch.nonzero(H == 1, as_tuple=False).t()
    H  = dglsp.spmatrix(coordinates)
    H = H.to(device)
    X = X.to(device)
    split_dict  = load_splits(split_path)
    train_batchloader = load_train(split_dict, args.bs, device, label=args.train_label)
    val_batchloader_pos = load_val(split_dict, args.bs, device, label="pos")
    val_batchloader_sns = load_val(split_dict, args.bs, device, label="sns")
    val_batchloader_mns = load_val(split_dict, args.bs, device, label="mns")
    val_batchloader_cns = load_val(split_dict, args.bs, device, label="cns")
    if args.train_label != 'mix_no_chem':
      #  val_batchloader_chem_neg = load_val(split_dict, args.bs, device, label="chem_neg")
        val_batchloader_chem_neg = load_val(split_dict, args.bs, device, label="chem_neg")
    test_batchloader_sns = load_test(split_dict, args.bs, device, label="sns")
    test_batchloader_mns = load_test(split_dict, args.bs, device, label="mns")
    test_batchloader_cns = load_test(split_dict, args.bs, device, label="cns")
    if args.train_label != 'mix_no_chem':
        test_batchloader_chem_neg = load_test(split_dict, args.bs, device, label="chem_neg")
    
    ## use the method from AHP, bartchloader to finish writing the loading of the training adn testing datasets
    ## training samples need to contains both training and testing
    #cls_layers = [args.dim_vertex, 128, 8, 1]
    cls_layers = [args.dim_vertex, 256, 16, 1]
    #print(f"H shape:{H.shape}")
    #print(f"X shape:{X.shape}")
   # pdb.set_trace()
    model = HGNN1(H.T, X.shape[1], args.dim_vertex, device).to(device)
    model_path = "./center_results/model_final.pkl"
    pretrained_gnn_model = load_pretrained_gnn_model(model_path, device, args_pretrained)
    #aggregator = MaxminAggregator(args.dim_vertex, cls_layers).to(device)
    aggregator = AttentionAggregator(args.dim_vertex, args.dim_gnn_embedding, 128, cls_layers, args.aggre_method).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(aggregator.parameters()), lr=args.lr)
    
    for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"Layer: {name} | Size: {param.size()} | Number of parameters: {param.numel()}")

    for name, param in aggregator.named_parameters():
            if param.requires_grad:
                print(f"Layer: {name} | Size: {param.size()} | Number of parameters: {param.numel()}")

    best_roc = 0
    best_epoch = 0 
    best_model = None
    best_aggr = None
    wandb.init(
        # set the wandb project where this run will be logged
        project=f"{args.dataset_name}_{args.aggre_method}_{args.encoding}_{args.cls_layers}_{args.dim_key_val}_{args.lr}_{args.epochs}",
        name=f"{args.loss_type}",
        # track hyperparameters and run metadata
        config={
        "learning_rate": args.lr,
        "dataset": args.dataset_name,
        "epochs": args.epochs,
        'cls_layers': args.cls_layers,
        'dim_key_val': args.dim_key_val,

        }
            )
    with tqdm.trange(args.epochs) as tq:
        for epoch in tq:

            loss_sum = 0
            count = 0
            train_pred, train_label = [], []
            i = 0
            while True:
                pos_hedges, pos_labels, is_last = train_batchloader.next()
                loss, pred, label = train(args, model, pretrained_gnn_model, optimizer, X, node_smiles_list, pos_hedges, pos_labels, aggregator, device)
                train_pred.extend(pred)
                train_label.extend(label)
        
                loss_sum += loss
                count += 1
                if is_last:
                    break
            train_pred = torch.stack(train_pred)
            train_pred = train_pred.squeeze()
            train_label = torch.round(torch.tensor(train_label))

            average_precision = AveragePrecision(task='binary')
            accuracy_metric = Accuracy(task='binary')
            train_ap = average_precision(torch.tensor(train_pred).cpu(), torch.tensor(train_label, dtype = torch.long).cpu())  
            train_acc = accuracy_metric(train_pred.cpu(), train_label.cpu())
            avg_loss = loss_sum / count
            print(f'train_ap:{train_ap}')
            print(f'loss:{avg_loss}') 

            val_pred_pos, total_label_pos, loss_pos = model_eval(model, pretrained_gnn_model, X, node_smiles_list, val_batchloader_pos, aggregator, device)
            val_pred_sns, total_label_sns, loss_sns = model_eval(model, pretrained_gnn_model, X, node_smiles_list, val_batchloader_sns, aggregator, device)
            output_sns = print_eval(torch.concat([val_pred_pos,val_pred_sns]).cpu(),torch.concat([total_label_pos,total_label_sns]).cpu(), (loss_pos + loss_sns).cpu()/2)
            #auc_roc_sns, ap_sns = utils.measure(total_label_pos+total_label_sns, val_pred_pos+val_pred_sns)
            f_log.write(output_sns)#f"{epoch} epoch, SNS : Val AP : {ap_sns} / AUROC : {auc_roc_sns}\n")

            val_pred_mns, total_label_mns, loss_mns = model_eval(model, pretrained_gnn_model, X, node_smiles_list, val_batchloader_mns, aggregator, device)
            output_mns = print_eval(torch.concat([val_pred_pos,val_pred_mns]).cpu(), torch.concat([total_label_pos,total_label_mns]).cpu(), (loss_pos + loss_mns).cpu()/2)
            f_log.write(output_mns)
            #auc_roc_mns, ap_mns = utils.measure(total_label_pos+total_label_mns, val_pred_pos+val_pred_mns)
            #f_log.write(f"{epoch} epoch, MNS : Val AP : {ap_mns} / AUROC : {auc_roc_mns}\n")
            val_pred_cns, total_label_cns, loss_cns = model_eval(model, pretrained_gnn_model, X, node_smiles_list, val_batchloader_cns, aggregator, device)
            output_cns = print_eval(torch.concat([val_pred_pos,val_pred_cns]).cpu(), torch.concat([total_label_pos,total_label_cns]).cpu(), (loss_pos + loss_cns).cpu()/2)
            f_log.write(output_cns)
            #auc_roc_cns, ap_cns = utils.measure(total_label_pos+total_label_cns, val_pred_pos+val_pred_cns)
            #f_log.write(f"{epoch} epoch, CNS : Val AP : {ap_cns} / AUROC : {auc_roc_cns}\n")
            if args.train_label != 'mix_no_chem':
                val_pred_chem_neg, total_label_chem_neg, loss_chem_neg = model_eval(model, pretrained_gnn_model, X, node_smiles_list, val_batchloader_chem_neg, aggregator, device)
                output_chem_neg = print_eval(torch.concat([val_pred_pos,val_pred_chem_neg]).cpu(), torch.concat([total_label_pos,total_label_chem_neg]).cpu(), (loss_pos + loss_chem_neg).cpu()/2)
                f_log.write(output_chem_neg)
                l = len(val_pred_pos)//4
                val_pred_all = torch.concat([val_pred_pos , val_pred_sns[0:l] , val_pred_mns[0:l] , val_pred_cns[0:l], val_pred_chem_neg[0:l]])
                total_label_all = torch.concat([total_label_pos , total_label_sns[0:l] , total_label_mns[0:l] , total_label_cns[0:l], total_label_chem_neg[0:l]])
                output_all = print_eval(val_pred_all.cpu(), total_label_all.cpu(), (loss_pos/2 + loss_cns/8 + loss_mns/8 + loss_sns/8 + loss_chem_neg/8).cpu())
                loss_all, accuracy_all, precision_all, recall_all, f1_all, specificity_all, auc_roc_all = get_eval(val_pred_all.cpu(), total_label_all.cpu(), (loss_pos/2 + loss_cns/8 + loss_mns/8 + loss_sns/8 + loss_chem_neg/8).cpu())
            else:
                l = len(val_pred_pos)//3
                val_pred_all = torch.concat([val_pred_pos , val_pred_sns[0:l] , val_pred_mns[0:l] , val_pred_cns[0:l]])
                total_label_all = torch.concat([total_label_pos , total_label_sns[0:l] , total_label_mns[0:l] , total_label_cns[0:l]])
                output_all = print_eval(val_pred_all.cpu(), total_label_all.cpu(), (loss_pos/2 + loss_cns/6 + loss_mns/6 + loss_sns/6 ).cpu())
                loss_all, accuracy_all, precision_all, recall_all, f1_all, specificity_all, auc_roc_all = get_eval(val_pred_all.cpu(), total_label_all.cpu(), (loss_pos/2 + loss_cns/6 + loss_mns/6 + loss_sns/6).cpu())
            f_log.write(output_all)
            f_log.flush()
            auc_roc = metrics.roc_auc_score(total_label_all.detach().cpu().numpy(), val_pred_all.detach().cpu().numpy())

            if best_roc < auc_roc:
                best_roc = auc_roc
                best_epoch=epoch
                best_model = model.state_dict()
                best_aggr = aggregator.state_dict()
                torch.save(model.state_dict(), f"{args.datapath}/checkpoints/{args.folder_name}_split{args.split}/model.pkt")
                torch.save(aggregator.state_dict(), f"{args.datapath}/checkpoints/{args.folder_name}_split{args.split}/Aggregator.pkt")
    
            wandb.log({
                            "train_loss": avg_loss,
                            "train_ap": train_ap,
                            "train_acc": train_acc,
                            "val_auc_roc": auc_roc_all,
                            "val_accuracy": accuracy_all,
                            "val_precision": precision_all,
                            "val_recall": recall_all,
                            "val_f1": f1_all,
                            "val_specificity": specificity_all,
                            
                        })
    wandb.finish()
    f_log.close()
    with open(f"{args.datapath}/checkpoints/{args.folder_name}_split{args.split}/best_epochs.logs", "a") as e_log:  
        e_log.write(f"best epochs: {best_epoch}, best roc: {best_roc}\n")

    with open(f"{args.datapath}/checkpoints/{args.folder_name}_split{args.split}/test_results.logs", "a") as r_log:
        model.load_state_dict(best_model)
        aggregator.load_state_dict(best_aggr)
        test_pred,test_label,loss = model_eval(model, pretrained_gnn_model, X, node_smiles_list, test_batchloader_mns, aggregator, device)
        output_test = 'mns: ' + print_eval(test_pred.cpu(), test_label.cpu(), loss.cpu())
        r_log.write(output_test)
        test_pred,test_label,loss = model_eval(model, pretrained_gnn_model, X, node_smiles_list, test_batchloader_sns, aggregator, device)
        output_test = 'sns: ' + print_eval(test_pred.cpu(), test_label.cpu(), loss.cpu())
        r_log.write(output_test)
        test_pred,test_label,loss = model_eval(model, pretrained_gnn_model, X, node_smiles_list, test_batchloader_cns, aggregator, device)
        output_test = 'cns: ' + print_eval(test_pred.cpu(), test_label.cpu(), loss.cpu())
        r_log.write(output_test)
        if args.train_label != 'mix_no_chem':
            test_pred,test_label,loss = model_eval(model, pretrained_gnn_model, X, node_smiles_list, test_batchloader_chem_neg, aggregator, device)
            output_test = 'chem_neg: ' + print_eval(test_pred.cpu(), test_label.cpu(), loss.cpu())
            r_log.write(output_test)
        