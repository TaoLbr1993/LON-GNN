import torch as th

from torch_geometric.nn.conv.gcn_conv import gcn_norm

def norm_edge_index(self, num_nodes, edge_index, norm=True, self_loop=True):
    edge_index, norm = gcn_norm(edge_index, None, num_nodes, add_self_loop=self_loop)
    return edge_index, norm
