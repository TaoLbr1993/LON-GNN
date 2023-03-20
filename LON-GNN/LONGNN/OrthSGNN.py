import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as thg
from torch_scatter import scatter_add
from torch_geometric.utils import get_laplacian
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes

from scipy.special import gamma as gammaFunc

from .MLP import MLP, MLPJ, Seq, ResBlock

def my_gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
                add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sparsesum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 1.0)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = th.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 1.0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

class LapMP(MessagePassing):
    def __init__(self, **kwargs):
        super(LapMP, self).__init__(aggr='add', **kwargs)
    
    def message(self, x_j, norm):
        return norm.view(-1, 1)*x_j


class OrthSGNN(nn.Module):
    # A base orth spectral GNN for ORTH basis
    # By Favard's theorem, each sequence of orthgonal polynomials can be represented in a recurrence relation: P_n(x)=(A_n*x+B_n)P_{n-1}(x)+C_n P_{n-2}(x-2).
    def __init__(self, dataset, args):
        super(OrthSGNN, self).__init__()
        data = dataset[0]
        if args.emb == 'mlp':
            self.mlp = MLP(data.x.shape[1], dataset.num_classes, args.hidden, args.dropout)
        elif args.emb == 'mlpj':
            self.mlp = MLPJ(data.x.shape[1], dataset.num_classes)
        elif args.emb == 'lin':
            self.mlp = nn.Linear(data.x.shape[1], dataset.num_classes)
        elif args.emb == 'res':
            self.mlp = Seq([
                nn.Linear(data.x.shape[1], dataset.num_classes),
                ResBlock(
                    nn.Sequential(nn.ReLU(inplace=True),
                    nn.Linear(dataset.num_classes, dataset.num_classes))
                )
            ])
        self.dropout = args.dropout
        self.dpr_con = args.dpr_con
        self.mp = LapMP()
        self.K = args.K 
        self.coef_upd = args.coef_upd 
        if self.coef_upd in ['pcd','none']:
            self.pcd_basealpha = args.alpha
            self.lap_coefs = nn.Parameter(th.Tensor([min(1/args.alpha, 1.)]*(args.K+1)))
        elif self.coef_upd == 'triangle':
            self.lap_coefs = nn.Parameter(th.Tensor([1.]+[0.]*args.K))

        self.edge_index, self.prop_norm = my_gcn_norm(data.edge_index, None, num_nodes=data.x.size(0), dtype=data.x.dtype, add_self_loops=False)
        self.edge_index = nn.Parameter(self.edge_index, requires_grad=False)
        self.prop_norm = nn.Parameter(self.prop_norm, requires_grad=False)
        
        self.norm_weights = th.Tensor([1.]*(args.K+1))
        if args.mulfilter:
            self.mf_weights = nn.Parameter(th.ones((1, args.K+1, dataset.num_classes)))
        else:
            self.mf_weights = nn.Parameter(th.ones(1, args.K+1, 1))

    # The coefficient functions return a float 
    def get_An(self, n):
        return 2

    def get_Bn(self, n):
        return 0

    def get_Cn(self, n):
        return -1

    # The init functions return a P0(L)*x and P1(L)*x
    def init_P0(self, x):
        return x

    def init_P1(self, x):
        return self.mp.propagate(x=x, edge_index=self.edge_index, norm=self.prop_norm)

    def init_coeffs(self, K, mode='Random', alpha=1.):
        if mode == 'Random':
            ret = th.rand(K+1)
        elif mode == 'PPR':
            ret = alpha*(1-alpha)**np.arange(K+1)
            ret[-1] = (1-alpha)**(K+1)
        elif mode == 'NPPR':
            ret = (alpha)**np.arange(K+1)
            ret = ret/np.sum(np.abs(ret))
        return ret

    def init_forward(self):
        pass

    def mulfilter_prop(self, coef_outs, mf_weights):
        '''
        Adapted from https://github.com/GraphPKU/JacobiConv
        '''
        ret = 0.
        us_outs = [out.unsqueeze(dim=1) for out in coef_outs]

        outs = th.cat(us_outs, dim=1)
        ret = outs*mf_weights
        ret = th.sum(ret, dim=1)
        return ret

    def coefs_trans(self, coefs, mode='none'):
        if mode == 'none':
            lap_coefs = self.pcd_basealpha*th.tanh(coefs)

            return coefs
        elif mode == 'pcd':
            # use 1., coefs[0:K-1]
            # seq: coefs[0:0]=1, coefs[0:1], prod(coefs[0:2]), ...
            #  
            lap_coefs = self.pcd_basealpha*th.tanh(coefs)
            return th.cumprod(lap_coefs, dim=0)

        elif mode == 'triangle':
            tri_coefs = th.Tensor([0.]*(self.K+1))
            tri_coefs[0] = coefs[0]
            for i in range(1,self.K):
                tri_coefs[i] = coefs[0]*th.prod(th.sin(coefs[1:i]))*th.cos(coefs[i])
            tri_coefs[self.K] = coefs[0]*th.prod(th.sin(coefs[1:self.K]))
            return tri_coefs

    @staticmethod
    def init_optim(model, args):
        if model.mf_weights is None:
            optimizer = th.optim.Adam([{
                'params': model.mlp.parameters(),
                'weight_decay': args.wd_mlp,
                'lr': args.lr_mlp
            },
            {
                'params': model.lap_coefs,
                'weight_decay': args.wd_lap,
                'lr': args.lr_lap
            }
            ])
        else:
            optimizer = th.optim.Adam([{
                'params': model.mlp.parameters(),
                'weight_decay': args.wd_mlp,
                'lr': args.lr_mlp
            },
            {
                'params': model.lap_coefs,
                'weight_decay': args.wd_lap,
                'lr': args.lr_lap
            },
            {
                'params': model.mf_weights,
                'weight_decay': args.wd_comb,
                'lr': args.lr_comb
            }
            ])
        return optimizer

    def forward(self, data):
        self.init_forward()
        x = data.x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.mlp(x)
        x = F.dropout(x, p=self.dpr_con, training=self.training)

        tmp_outs = [self.init_P0(x), self.init_P1(x)]
        for i in range(2, self.K+1):
            tmp_part1 = self.get_Bn(i)*tmp_outs[-1] + self.get_Cn(i)*tmp_outs[-2]
            tmp_part2 = self.get_An(i) * self.mp.propagate(self.edge_index, x=tmp_outs[-1], norm=self.prop_norm)
            tmp = tmp_part1 + tmp_part2
            tmp_outs.append(tmp)
        coef_outs = []

        lap_coefs = self.coefs_trans(self.lap_coefs, mode=self.coef_upd)

        coef_outs.append(tmp_outs[0]/self.norm_weights[0])
        for i in range(1, self.K+1):
            coef_out = lap_coefs[i-1]/self.norm_weights[i]*tmp_outs[i]
            coef_outs.append(coef_out)
        retx = self.mulfilter_prop(coef_outs, self.mf_weights)
        return F.log_softmax(retx, dim=-1)


class JacobiSGNN(OrthSGNN):
    def __init__(self, dataset, args):
        super(JacobiSGNN, self).__init__(dataset, args)
        self.a = nn.Parameter(th.tensor(args.jacob_a), requires_grad=False)
        self.b = nn.Parameter(th.tensor(args.jacob_b), requires_grad=False)

    def init_P0(self, x):
        return x
    
    def init_P1(self, x):
        c0 = (self.a-self.b)/2.
        c1 = (self.a+self.b+2)/2.
        Lx = self.mp.propagate(self.edge_index, x=x, norm=self.prop_norm)
        return c0*x + c1*Lx

    def get_An(self, n):
        nab = 2*n+self.a+self.b
        numerator = nab*(nab-1)*(nab-2)
        denominator = 2*n*(nab-n)*(nab-2)
        return numerator/denominator

    def get_Bn(self, n):
        nab = 2*n+self.a+self.b
        numerator = (nab-1)*(self.a*self.a-self.b*self.b)
        denominator = 2*n*(nab-n)*(nab-2)
        return -numerator/denominator

    def get_Cn(self, n):
        nab = 2*n+self.a+self.b
        numerator = 2*(n+self.a-1)*(n+self.b-1)*nab
        denominator = 2*n*(nab-n)*(nab-2)
        return -numerator/denominator

class JacobiSGNNS(JacobiSGNN):
    def __init__(self, dataset, args):
        super(JacobiSGNN, self).__init__(dataset, args)
        self.a = nn.Parameter(th.tensor(args.jacob_a), requires_grad=True)
        self.b = nn.Parameter(th.tensor(args.jacob_b), requires_grad=True)

    @staticmethod
    def init_optim(model, args):
        optimizer = th.optim.Adam([{
                'params': model.mlp.parameters(),
                'weight_decay': args.wd_mlp,
                'lr': args.lr_mlp
            },
            {
                'params': model.lap_coefs,
                'weight_decay': args.wd_lap,
                'lr': args.lr_lap
            },
            {
                'params': [model.a,model.b],
                'weight_decay': args.wd_ab,
                'lr': args.lr_ab
            },
            {
                'params': model.mf_weights,
                'weight_decay': args.wd_comb,
                'lr': args.lr_comb
            }
            ])
        return optimizer

class StdOrthSGNN(OrthSGNN):
    def __init__(self, dataset, args):
        super(StdOrthSGNN, self).__init__(dataset, args)

        # init norm weight
        self.norm_weights = self.get_norm_weight_formula()

    def get_norm_weight_formula(self):
        # chebyshev
        ret = [np.sqrt(np.pi)]+[np.sqrt(np.pi/2)]*self.K
        return ret


class StdJacobiSGNN(StdOrthSGNN):
    def __init__(self, dataset, args):
        super(StdJacobiSGNN, self).__init__(dataset, args)
        self.a = nn.Parameter(th.tensor(args.jacob_a), requires_grad=False)
        self.b = nn.Parameter(th.tensor(args.jacob_b), requires_grad=False)
        self.norm_weights = self.get_norm_weight_formula()

    def init_P1(self, x):
        c0 = (self.a-self.b)/2.
        c1 = (self.a+self.b+2)/2.
        Lx = self.mp.propagate(self.edge_index, x=x, norm=self.prop_norm)
        return c0*x + c1*Lx

    def get_An(self, n):
        nab = 2*n+self.a+self.b
        numerator = nab*(nab-1)*(nab-2)
        denominator = 2*n*(nab-n)*(nab-2)
        return numerator/denominator

    def get_Bn(self, n):
        nab = 2*n+self.a+self.b
        numerator = (nab-1)*(self.a*self.a-self.b*self.b)
        denominator = 2*n*(nab-n)*(nab-2)
        return numerator/denominator
    
    def get_Cn(self, n):
        nab = 2*n+self.a+self.b
        numerator = 2*(n+self.a-1)*(n+self.b-1)*nab
        denominator = 2*n*(nab-n)*(nab-2)
        return -numerator/denominator
            
    def gammaFunc_torch(self, x):
        return th.exp(th.special.gammaln(x))

    def get_norm_weight_formula(self):
        if not hasattr(self, 'a'):
            return None
        ret = []
        for i in range(self.K+1):
            term1 = th.pow(2, self.a+self.b+1)/(2*i+self.a+self.b+1)
            term2 = self.gammaFunc_torch(i+self.a+1)/self.gammaFunc_torch(i+self.a+self.b+1)
            term3 = self.gammaFunc_torch(i+self.b+1)/self.gammaFunc_torch(th.tensor(i+1))
            ret.append(th.sqrt(term1*term2*term3))
        return ret

    def get_norm_weight_formula_np(self):
        if not hasattr(self, 'a'):
            return None
        a = self.a.item()
        b = self.b.item()
        ret = []
        for i in range(self.K+1):
            term1 = np.power(2, a+b+1)/(2*i+a+b+1)
            term2 = gammaFunc(i+a+1)/gammaFunc(i+a+b+1)
            term3 = gammaFunc(i+b+1)/gammaFunc(i+1)
            ret.append(np.sqrt(term1*term2*term3))
        return ret
    
class StdJacobiSGNNS(StdJacobiSGNN):
    def __init__(self, dataset, args):
        super(StdJacobiSGNN, self).__init__(dataset, args)
        self.a = nn.Parameter(th.tensor(args.jacob_a), requires_grad=True)
        self.b = nn.Parameter(th.tensor(args.jacob_b), requires_grad=True)

    def init_forward(self):
        self.norm_weights = self.get_norm_weight_formula()

    @staticmethod
    def init_optim(model, args):
        optimizer = th.optim.Adam([{
                'params': model.mlp.parameters(),
                'weight_decay': args.wd_mlp,
                'lr': args.lr_mlp
            },
            {
                'params': model.lap_coefs,
                'weight_decay': args.wd_lap,
                'lr': args.lr_lap
            },
            {
                'params': [model.a,model.b],
                'weight_decay': args.wd_ab,
                'lr': args.lr_ab
            },
            {
                'params': model.mf_weights,
                'weight_decay': args.wd_comb,
                'lr': args.lr_comb
            }
            ])
        return optimizer
