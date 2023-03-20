import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as thg
from torch_geometric.utils import get_laplacian
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm

class MLP(nn.Module):
    def __init__(self, in_feat, out_feat, hidden, dropout=0):
        super(MLP, self).__init__()
        self.lin1 = nn.Linear(in_feat, hidden)
        self.lin2 = nn.Linear(hidden, out_feat)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return x

class LapMP(MessagePassing):
    def __init__(self, **kwargs):
        super(LapMP, self).__init__(aggr='add', **kwargs)
    
    def message(self, x_j, norm):
        return norm.view(-1, 1)*x_j


class OrthSGNN(nn.Module):
    # A base orth spectral GNN for ORTH basis
    # By Favard's theorem, each sequence of orthgonal polynomials can be represented in a recurrence relation: P_n(x)=(A_n*x+B_n)P_{n-1}(x)+C_n P_{n-2}(x-2).
    def __init__(self, args):
        super(OrthSGNN, self).__init__()
        self.mp = LapMP()   
        self.K = 10
        self.coef_upd = args.coef_upd 
        if self.coef_upd in ['pcd','none']:
            self.pcd_basealpha = args.alpha
            self.lap_coefs = nn.Parameter(th.Tensor([min(1/args.alpha, 1.)]*(self.K+1)), requires_grad=False)
        elif self.coef_upd == 'triangle':
            self.lap_coefs = nn.Parameter(th.Tensor([1.]+[0.]*self.K))

        self.norm_weights = th.Tensor([1.]*(self.K+1))
        self.mf_weights = nn.Parameter(th.ones((1, self.K+1, 1)), requires_grad=True)
        self.ifTrainable = False

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

    def init_P1(self, x, edge_index, prop_norm):
        return self.mp.propagate(x=x, edge_index=edge_index, norm=prop_norm)

    def init_coeffs(self, K, mode='Random', alpha=1.):
        if mode == 'Random':
            ret = th.rand(K+1)
            # ret = ret/ret.sum(ret)
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
            lap_coefs = self.pcd_basealpha*th.tanh(coefs)
            return th.cumprod(lap_coefs, dim=0)

        elif mode == 'triangle':
            tri_coefs = th.Tensor([0.]*(self.K+1))
            tri_coefs[0] = coefs[0]
            for i in range(1,self.K):
                tri_coefs[i] = coefs[0]*th.prod(th.sin(coefs[1:i]))*th.cos(coefs[i])
            tri_coefs[self.K] = coefs[0]*th.prod(th.sin(coefs[1:self.K]))
            return tri_coefs

    def forward(self, data):
        self.init_forward()
        edge_index, prop_norm = gcn_norm(data.edge_index, None, num_nodes=data.x.size(0), dtype=data.x.dtype, add_self_loops=False)

        x = data.x_tmp
        tmp_outs = [self.init_P0(x), self.init_P1(x, edge_index, prop_norm)]

        for i in range(2, self.K+1):
            tmp_part1 = self.get_Bn(i)*tmp_outs[-1] + self.get_Cn(i)*tmp_outs[-2]
            tmp_part2 = self.get_An(i) * self.mp.propagate(edge_index, x=tmp_outs[-1], norm=prop_norm)
            tmp = tmp_part1 + tmp_part2
            tmp_outs.append(tmp)
        coef_outs = []


        lap_coefs = self.coefs_trans(self.lap_coefs, mode=self.coef_upd)
        coef_outs.append(tmp_outs[0]/self.norm_weights[0])
        for i in range(1, self.K+1):
            coef_out = lap_coefs[i-1]/self.norm_weights[i]*tmp_outs[i]
            coef_outs.append(coef_out)

        retx = self.mulfilter_prop(coef_outs, self.mf_weights)
        return retx


class JacobiSGNN(OrthSGNN):
    def __init__(self, args):
        super(JacobiSGNN, self).__init__(args)
        self.a = nn.Parameter(th.tensor(args.jacob_a), requires_grad=False)
        self.b = nn.Parameter(th.tensor(args.jacob_b), requires_grad=False)
        self.ifTrainable = False

    def init_P0(self, x):
        return x
    
    def init_P1(self, x, edge_index, prop_norm):
        if False:
            a = th.clamp(self.a, -1.0+0.00001, 10.0)
            b = th.clamp(self.b, -1.0+0.00001, 10.0)
        else:
            a = self.a
            b = self.b
        c0 = (a-b)/2.
        c1 = (a+b+2)/2.
        Lx = self.mp.propagate(edge_index, x=x, norm=prop_norm)
        return c0*x + c1*Lx

    def get_An(self, n):
        if False:
            a = th.clamp(self.a, -1.0+0.00001, 10.0)
            b = th.clamp(self.b, -1.0+0.00001, 10.0)
        else:
            a = self.a
            b = self.b
        nab = 2*n+a+b
        numerator = nab*(nab-1)*(nab-2)
        denominator = 2*n*(nab-n)*(nab-2)
        return numerator/denominator

    def get_Bn(self, n):
        if False:
            a = th.clamp(self.a, -1.0+0.00001, 10.0)
            b = th.clamp(self.b, -1.0+0.00001, 10.0)
        else:
            a = self.a
            b = self.b
        nab = 2*n+a+b
        numerator = (nab-1)*(a*a-b*b)
        denominator = 2*n*(nab-n)*(nab-2)
        return numerator/denominator

    def get_Cn(self, n):
        if False:
            a = th.clamp(self.a, -1.0+0.00001, 10.0)
            b = th.clamp(self.b, -1.0+0.00001, 10.0)
        else:
            a = self.a
            b = self.b
        nab = 2*n+a+b
        numerator = 2*(n+a-1)*(n+b-1)*nab
        denominator = 2*n*(nab-n)*(nab-2)
        return -numerator/denominator


class JacobiSGNNS(JacobiSGNN):
    def __init__(self, args):
        super(JacobiSGNN, self).__init__(args)
        self.a = nn.Parameter(th.tensor(args.jacob_a), requires_grad=True)
        self.b = nn.Parameter(th.tensor(args.jacob_b), requires_grad=True)
        self.ifTrainable = True

class StdOrthSGNN(OrthSGNN):
    def __init__(self, args):
        super(StdOrthSGNN, self).__init__(args)

    def get_norm_weight_formula(self):
        ret = [np.sqrt(np.pi)]+[np.sqrt(np.pi/2)]*self.K
        return ret


class StdJacobiSGNN(StdOrthSGNN):
    def __init__(self, args):
        self.isTrainable = False
        super(StdJacobiSGNN, self).__init__(args)
        self.a = nn.Parameter(th.tensor(args.jacob_a), requires_grad=False)
        self.b = nn.Parameter(th.tensor(args.jacob_b), requires_grad=False)
        self.norm_weights = self.get_norm_weight_formula()

    def init_P1(self, x, edge_index, prop_norm):
        if self.isTrainable:
            a = th.clamp(self.a, -1.0+0.00001, 10.0)
            b = th.clamp(self.b, -1.0+0.00001, 10.0)
        else:
            a = self.a
            b = self.b
        if a+b <= -1.0:
            gap = (-a-b-1.0+0.001)
            a = a + gap/2
            b = b + gap/2

        c0 = (a-b)/2.
        c1 = (a+b+2)/2.
        Lx = self.mp.propagate(edge_index, x=x, norm=prop_norm)
        return c0*x + c1*Lx

    def get_An(self, n):
        if self.isTrainable:
            a = th.clamp(self.a, -1.0+0.00001, 10.0)
            b = th.clamp(self.b, -1.0+0.00001, 10.0)
        else:
            a = self.a
            b = self.b
        if a+b <= -1.0:
            gap = (-a-b-1.0+0.0001)
            a = a + gap/2
            b = b + gap/2
        nab = 2*n+a+b
        numerator = nab*(nab-1)*(nab-2)
        denominator = 2*n*(nab-n)*(nab-2)
        return numerator/denominator

    def get_Bn(self, n):
        if self.isTrainable:
            a = th.clamp(self.a, -1.0+0.00001, 10.0)
            b = th.clamp(self.b, -1.0+0.00001, 10.0)
        else:
            a = self.a
            b = self.b
        if a+b <= -1.0:
            gap = (-a-b-1.0+0.0001)
            a = a + gap/2
            b = b + gap/2
        nab = 2*n+a+b
        numerator = (nab-1)*(a*a-b*b)
        denominator = 2*n*(nab-n)*(nab-2)
        return numerator/denominator
    
    def get_Cn(self, n):
        if self.isTrainable:
            a = th.clamp(self.a, -1.0+0.00001, 10.0)
            b = th.clamp(self.b, -1.0+0.00001, 10.0)
        else:
            a = self.a
            b = self.b
        if a+b <= -1.0:
            gap = (-a-b-1.0+0.0001)
            a = a + gap/2
            b = b + gap/2
        nab = 2*n+a+b
        numerator = 2*(n+a-1)*(n+b-1)*nab
        denominator = 2*n*(nab-n)*(nab-2)
        return -numerator/denominator
            
    def gammaFunc_torch(self, x):
        return th.exp(th.special.gammaln(x))

    def get_norm_weight_formula(self):
        if self.isTrainable:
            a = th.clamp(self.a, -1.0+0.00001, 10.0)
            b = th.clamp(self.b, -1.0+0.00001, 10.0)
        else:
            a = self.a
            b = self.b
        if a+b <= -1.0:
            gap = (-a-b-1.0+0.0001)
            a = a + gap/2
            b = b + gap/2
        if not hasattr(self, 'a'):
            return None
        ret = []
        for i in range(self.K+1):
            term1 = th.pow(2, a+b+1)/(2*i+a+b+1)
            term2 = self.gammaFunc_torch(i+a+1)/self.gammaFunc_torch(i+a+b+1)
            term3 = self.gammaFunc_torch(i+b+1)/self.gammaFunc_torch(th.tensor(i+1))
            ret.append(th.sqrt(term1*term2*term3))
        return ret

class StdJacobiSGNNS(StdJacobiSGNN):
    def __init__(self, args):
        self.isTrainable = True

        super(StdJacobiSGNN, self).__init__(args)
        self.a = nn.Parameter(th.tensor(args.jacob_a), requires_grad=True)
        self.b = nn.Parameter(th.tensor(args.jacob_b), requires_grad=True)
        
    def init_forward(self):
        self.norm_weights = self.get_norm_weight_formula()

