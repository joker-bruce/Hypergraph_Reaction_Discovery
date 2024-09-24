# Install required packages.
import os
import torch
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

H = dglsp.spmatrix(
    torch.LongTensor([[0, 1, 2, 2, 2, 2, 3, 4, 5, 5, 5, 5, 6, 7, 7, 8, 8, 9, 9, 10],
                      [0, 0, 0, 1, 3, 4, 2, 1, 0, 2, 3, 4, 2, 1, 3, 1, 3, 2, 4, 4]])
)

print(H.to_dense())

node_degrees = H.sum(1)
print("Node degrees", node_degrees)

hyperedge_degrees = H.sum(0)
print("Hyperedge degrees", hyperedge_degrees)


import dgl.sparse as dglsp
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from dgl.data import CoraGraphDataset
from torchmetrics.functional import accuracy


class HGNN(nn.Module):
    def __init__(self, H, in_size, out_size, hidden_dims=16):
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
        B = dglsp.identity((n_edges, n_edges))
        # Compute Laplacian from the equation above.
        self.L = D_v_invsqrt @ H @ B @ D_e_inv @ H.T @ D_v_invsqrt

    def forward(self, X):
        X = self.L @ self.W1(self.dropout(X))
        X = F.relu(X)
        X = self.L @ self.W2(self.dropout(X))
        return X
    

def load_data():
    dataset = CoraGraphDataset()

    graph = dataset[0]
    indices = torch.stack(graph.edges())
    H = dglsp.spmatrix(indices)
    H = H + dglsp.identity(H.shape)

    X = graph.ndata["feat"]
    Y = graph.ndata["label"]
    train_mask = graph.ndata["train_mask"]
    val_mask = graph.ndata["val_mask"]
    test_mask = graph.ndata["test_mask"]
    return H, X, Y, dataset.num_classes, train_mask, val_mask, test_mask

def train(model, optimizer, X, Y, train_mask):
    model.train()
    Y_hat = model(X)
    loss = F.cross_entropy(Y_hat[train_mask], Y[train_mask])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def evaluate(model, X, Y, val_mask, test_mask, num_classes):
    model.eval()
    Y_hat = model(X)
    val_acc = accuracy(
        Y_hat[val_mask], Y[val_mask], task="multiclass", num_classes=num_classes
    )
    test_acc = accuracy(
        Y_hat[test_mask],
        Y[test_mask],
        task="multiclass",
        num_classes=num_classes,
    )
    return val_acc, test_acc


H, X, Y, num_classes, train_mask, val_mask, test_mask = load_data()
model = HGNN(H, X.shape[1], num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

with tqdm.trange(500) as tq:
    for epoch in tq:
        train(model, optimizer, X, Y, train_mask)
        val_acc, test_acc = evaluate(
            model, X, Y, val_mask, test_mask, num_classes
        )
        tq.set_postfix(
            {
                "Val acc": f"{val_acc:.5f}",
                "Test acc": f"{test_acc:.5f}",
            },
            refresh=False,
        )

print(f"Test acc: {test_acc:.3f}")