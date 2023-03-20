import os
from functools import partial
import argparse
import numpy as np
import torch as th
import torch.nn.functional as F
import torch_geometric as thg
import matplotlib.pyplot as plt

from data import load_dataset
from utils import RAND_SEED, init_seed, rand_split
from models import *
from LONGNN import *
from search_params import SEARCH_PARAMS

def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    logits = model(data)
    loss = F.nll_loss(logits[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

@torch.no_grad()
def test(model, data):
    def logits_to_loss_accu(logits, y):
        loss = F.nll_loss(logits, y)
        pred = logits.max(1)[1]
        accu = (pred==y).sum().item()/float(y.shape[0])
        return loss, accu
    model.eval()
    logits = model(data)
    train_loss, train_accu = logits_to_loss_accu(logits[data.train_mask], data.y[data.train_mask])
    val_loss, val_accu = logits_to_loss_accu(logits[data.val_mask], data.y[data.val_mask])
    test_loss, test_accu = logits_to_loss_accu(logits[data.test_mask], data.y[data.test_mask])
    return train_loss, train_accu, val_loss, val_accu, test_loss, test_accu


def single_run(MODEL, args, dataset):
    # split data
    data = dataset[0]
    data = rand_split(data, args.train_rate, args.val_rate, dataset.num_classes)

    model = MODEL(dataset, args)

    # update to device
    device = torch.device('cuda:{}'.format(args.gpu_id) if th.cuda.is_available() else 'cpu')
    model.to(device)
    data = data.to(device)

    # init optimizer
    if hasattr(model, 'init_optim'):
        optimizer = model.init_optim(model, args)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_mlp, weight_decay=args.wd_mlp)
    
    val_loss_list = []
    val_acc_list = []
    test_acc_list = []
    best_val_loss = float('inf')
    best_val_accu = 0.
    eval_accu = 0.
    no_improvement_cnt = 0

    for epoch in range(args.epochs):
        train(model, optimizer, data)
        train_loss, train_accu, val_loss, val_accu, test_loss, test_accu = test(model, data)
        val_loss_list.append(val_loss.item())
        val_acc_list.append(val_accu)
        test_acc_list.append(test_accu)

        # early stop
        if args.early_stop_cond == 0:
            # early stop strategy 1:
            # loss > the latest 200 average loss
            # print(train_accu)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_accu = val_accu
                eval_accu = test_accu
            if epoch > args.early_stop:
                avg_loss = np.mean(val_loss_list[-(args.early_stop+1):-1])
                if val_loss > avg_loss:
                    # print('early stop at eph {}'.format(epoch))
                    break
        elif args.early_stop_cond == 1:
            if val_accu >= best_val_accu:
                no_improvement_cnt = 0
                best_val_accu = val_accu
                eval_accu = test_accu
            else:
                no_improvement_cnt += 1
            if no_improvement_cnt > args.early_stop:
                break
        else:
            raise NotImplementedError("Invalid early stop condition!")
                
        if args.repeat == 1 and args.eph_filter > 0:
            if epoch % args.eph_filter == 0:
                xs, filters = model.draw_filter()
                plt.plot(xs, filters)
                plt.ylim((-10,10))
                plt.title('epoch {}'.format(epoch))
                plt.show()   
    
    return eval_accu, best_val_accu

def run(MODEL, args, dataset, output=True):
    test_accus = []
    val_accus = []
    for rp in range(args.repeat):
        init_seed(RAND_SEED[rp])
        test_accu, val_accu = single_run(MODEL, args, dataset)
        test_accus.append(test_accu)
        val_accus.append(val_accu)

    # print(test_accu)
    avg_test_accu = np.mean(test_accus)*100
    avg_val_accu = np.mean(val_accus)*100
    std_accu = np.std(test_accus)*100
    if output:
        print(f'Dataset {args.dataset}: test accu: {avg_test_accu}, std: {std_accu}, val_accu: {avg_val_accu}')
    return avg_test_accu, std_accu, avg_val_accu


def param_search(trial, MODEL, args, dataset):
    params = SEARCH_PARAMS[args.model]
    for key in params.keys():
        setattr(args, key, trial.suggest_categorical(key, params[key]))
    
    _, _, accu = run(MODEL, args, dataset, output=False)
    return accu


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--dataset', type=str, default='cornell')
    parser.add_argument('--train_rate', type=float, default=0.6)
    parser.add_argument('--val_rate', type=float, default=0.2)
    # training arguments
    parser.add_argument('--model', type=str, default='GPRGNN')
    parser.add_argument('--ds_fp', type=str, default=None)
    parser.add_argument('--repeat', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr_mlp', type=float, default=0.01)
    parser.add_argument('--lr_lap', type=float, default=0.01)
    parser.add_argument('--lr_comb', type=float, default=0.01)
    parser.add_argument('--lr_ab', type=float, default=0.01)
    parser.add_argument('--early_stop', type=int, default=200)
    parser.add_argument('--early_stop_cond', type=int, default=1)
    parser.add_argument('--stop_mode', type=str, choices=['loss', 'filter'], default='loss')
    parser.add_argument('--gpu_id', type=int, default=0)
    # Spectral GNN arguments
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--emb', type=str, default='lin', choices=['lin','mlp','mlpj','res'])
    parser.add_argument('--wd_mlp', type=float, default=0.0005)
    parser.add_argument('--wd_lap', type=float, default=0.0005)
    parser.add_argument('--wd_comb', type=float, default=0.0005)
    parser.add_argument('--wd_ab', type=float, default=0.0)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--dpr_con', type=float, default=0.5)
    parser.add_argument('--mulfilter', type=bool, default=True)
    parser.add_argument('--coef_upd', type=str, default='none')
    parser.add_argument('--jacob_a', type=float, default=0.5)
    parser.add_argument('--jacob_b', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=1.)

    # baseline arguments
    parser.add_argument('--heads', type=int, default=2)
    parser.add_argument('--output_heads', type=int, default=2)
    # debugging arguments
    parser.add_argument('--alph_curve', action='store_true')
    parser.add_argument('--show_filter', action='store_true')
    parser.add_argument('--eph_filter', type=int, default=-1)

    # param search arguments
    parser.add_argument('--param_search', action='store_true')
    parser.add_argument('--storage', type=str, default=None)
    parser.add_argument('--n_trials', type=int, default=400)
    args = parser.parse_args()
    dataset = load_dataset(args.dataset, args.ds_fp)
    # init model and args
    if args.model == 'GCN':
        MODEL = GCN
    elif args.model == 'ChebNet':
        MODEL = ChebNet
    elif args.model == 'GAT':
        MODEL = GAT_Net
    elif args.model == 'APPNP':
        MODEL = APPNP_Net
    elif args.model == 'GPRGNN':
        MODEL = GPRGNN
        args.Init = 'PPR'
    elif args.model == 'ChebySGNN':
        MODEL = OrthSGNN
    elif args.model == 'BernNet':
        MODEL = BernNet
    elif args.model == 'JacobiSGNN':
        MODEL = JacobiSGNN
    elif args.model == 'JacobiSGNNS':
        MODEL = JacobiSGNNS
    elif args.model == 'StdJacobiSGNN':
        MODEL = StdJacobiSGNN
    elif args.model == 'StdJacobiSGNNS':
        MODEL = StdJacobiSGNNS

    if args.param_search:
        args.repeat = 3
        import optuna
        if args.storage:
            abspath = os.path.abspath(__file__)
            storage_path = os.path.join(os.path.dirname(abspath), args.storage)
            if not os.path.exists(storage_path):
                os.mkdir(storage_path)
            dbfp = "sqlite:///"+str(os.path.join(storage_path,'res_{}_{}.sqlite').format(args.model, args.dataset))
            print(dbfp)
            storage = optuna.storages.RDBStorage(url=dbfp, engine_kwargs={"connect_args": {"timeout": 800}})
        else:
            storage = None
        sampler = None
        study = optuna.create_study(direction='maximize', study_name = 'opt_'+args.model+'_'+args.dataset, load_if_exists=True, storage=storage, sampler=sampler)
        study.optimize(partial(param_search, MODEL=MODEL, args=args, dataset=dataset), n_trials=args.n_trials)
        print(study.best_params)
        print(study.best_value)
    else:
        print(args)
        run(MODEL, args, dataset)
