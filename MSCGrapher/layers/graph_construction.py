import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import LEConv
from torch_geometric.utils import degree


class CausalLearning(nn.Module):
    def __init__(self, channels, causal_ratio, in_channel):
        super(CausalLearning, self).__init__()
        self.conv1 = LEConv(in_channels=in_channel, out_channels=channels)
        self.conv2 = LEConv(in_channels=channels, out_channels=channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels * 2, channels * 4),
            nn.ReLU(),
            nn.Linear(channels * 4, 1)
        )
        self.ratio = causal_ratio

    def forward(self, data):
        # batch_norm
        x = F.relu(self.conv1(data.x, data.edge_index, data.edge_attr.view(-1)))
        x = self.conv2(x, data.edge_index, data.edge_attr.view(-1))

        # row, col = data.edge_index
        # edge_rep = torch.cat([x[row], x[col]], dim=-1)
        # edge_score = self.mlp(edge_rep).view(-1)

        row, col = data.edge_index
        edge_rep = torch.cat([x[row], x[col]], dim=-1)
        edge_score = self.mlp(edge_rep).view(-1)

        (causal_edge_index, causal_edge_attr, causal_edge_weight), \
            (conf_edge_index, conf_edge_attr, conf_edge_weight) = split_graph(data, edge_score, self.ratio)

        causal_x, causal_edge_index, causal_batch, _ = relabel(x, causal_edge_index, data.batch)
        conf_x, conf_edge_index, conf_batch, _ = relabel(x, conf_edge_index, data.batch)

        return (causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch), \
            (conf_x, conf_edge_index, conf_edge_attr, conf_edge_weight, conf_batch), \
            edge_score


def split_batch(g):
    split = degree(g.batch[g.edge_index[0]], dtype=torch.long).tolist()
    edge_indices = torch.split(g.edge_index, split, dim=1)
    num_nodes = degree(g.batch, dtype=torch.long)
    cum_nodes = torch.cat([g.batch.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]])
    num_edges = torch.tensor([e.size(1) for e in edge_indices], dtype=torch.long).to(g.x.device)
    cum_edges = torch.cat([g.batch.new_zeros(1), num_edges.cumsum(dim=0)[:-1]])

    return edge_indices, num_nodes, cum_nodes, num_edges, cum_edges


def split_graph(data, edge_score, ratio):
    causal_edge_index = torch.LongTensor([[], []]).to(data.x.device)
    causal_edge_weight = torch.tensor([]).to(data.x.device)
    causal_edge_attr = torch.tensor([]).to(data.x.device)
    conf_edge_index = torch.LongTensor([[], []]).to(data.x.device)
    conf_edge_weight = torch.tensor([]).to(data.x.device)
    conf_edge_attr = torch.tensor([]).to(data.x.device)

    edge_indices, _, _, num_edges, cum_edges = split_batch(data)
    for edge_index, N, C in zip(edge_indices, num_edges, cum_edges):
        n_reserve = int(ratio * N)
        edge_attr = data.edge_attr[C:C + N]
        single_mask = edge_score[C:C + N]
        single_mask_detach = edge_score[C:C + N].detach().cpu().numpy()
        rank = np.argpartition(-single_mask_detach, n_reserve)
        idx_reserve, idx_drop = rank[:n_reserve], rank[n_reserve:]

        causal_edge_index = torch.cat([causal_edge_index, edge_index[:, idx_reserve]], dim=1)
        conf_edge_index = torch.cat([conf_edge_index, edge_index[:, idx_drop]], dim=1)

        causal_edge_weight = torch.cat([causal_edge_weight, single_mask[idx_reserve]])
        conf_edge_weight = torch.cat([conf_edge_weight, -1 * single_mask[idx_drop]])

        causal_edge_attr = torch.cat([causal_edge_attr, edge_attr[idx_reserve]])
        conf_edge_attr = torch.cat([conf_edge_attr, edge_attr[idx_drop]])
    return (causal_edge_index, causal_edge_attr, causal_edge_weight), \
        (conf_edge_index, conf_edge_attr, conf_edge_weight)


def relabel(x, edge_index, batch, pos=None):
    num_nodes = x.size(0)
    sub_nodes = torch.unique(edge_index)
    x = x[sub_nodes]
    batch = batch[sub_nodes]
    row, col = edge_index
    # remapping the nodes in the explanatory subgraph to new ids.
    node_idx = row.new_full((num_nodes,), -1)
    node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=row.device)
    edge_index = node_idx[edge_index]
    if pos is not None:
        pos = pos[sub_nodes]
    return x, edge_index, batch, pos