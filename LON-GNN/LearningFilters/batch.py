import os
import argparse

BASELINES = ['ChebNet','BernNet','GcnNet','GatNet','ARMANet','GPRNet', 'JacobiSGNN', 'JacobiSGNNS', 'StdJacobiSGNN', 'StdJacobiSGNNS']
OPT_MODELS = ['JacobiSGNN', 'JacobiSGNNS', 'StdJacobiSGNN', 'StdJacobiSGNNS']

DATASETS = ['low','high','band','rejection','comb','low_band']

# please keep this variable same as that in batch_run.sh
FIX_ARGS = ''

def get_best_param_res(fp):
    with open(fp, 'r') as f:
        params = f.readline()
        if params == '':
            return None, None
        if params.startswith('sqlite'):
            params = f.readline()
        params = ast.literal_eval(params)
        line = f.readline()
        best_val_acc = float(line)
        return params, best_val_acc

def gen_searchM_cmds(args, dataset=None, model=None):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    if dataset is None:
        datasets = DATASETS
    else: datasets = [dataset]
    if model is None:
        models = OPT_MODELS
    else: models = [model]
    cmds = []
    for ds in datasets:
        for model in models:
            cmd = "python training.py --filter_type {} --net {}".format(ds, model)
            cmd = cmd + ' ' + '--param_search'
            if args.gpu_id >=0:
                cmd = cmd + ' --gpu_id {}'.format(args.gpu_id)
            if FIX_ARGS != '':
                cmd = cmd + ' ' + FIX_ARGS
            if args.redir:
                res_name = 'test_{}_{}'.format(model, ds)
                err_fp = '{}.err'.format(res_name)
                out_fp = '{}.out'.format(res_name)
                cmd = cmd + ' >{} 2>{}'.format(out_fp, err_fp)
            cmds.append(cmd)
            
    return cmds


def gen_baseline_cmds_dataset(args, ds):
    cmds = []
    for model in BASELINES:
        cmd = "python training.py --filter_type {} --net {}".format(ds, model)
        if args.gpu_id >= 0:
            cmd  = cmd + ' --gpu_id {}'.format(args.gpu_id)
        if FIX_ARGS != '':
            cmd = cmd + ' ' + FIX_ARGS
        if args.redir:
            fp_name = 'test_{}_{}'.format(ds, model)
            cmd  = cmd + ' > {}.out 2>{}.err'.format(fp_name, fp_name)
        cmds.append(cmd)
    return cmds

def gen_baseline_cmds(args, ds):
    if ds is None:
        datasets = DATASETS
        cmds = []
        for ds in datasets:
            tmp_cmds = gen_baseline_cmds_dataset( args, ds)
            cmds = cmds + tmp_cmds
        return cmds
    else:
        return gen_baseline_cmds_dataset(args, ds)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--gpu_id', type=int, default=-1)
    parser.add_argument('--fp', type=str, default='./test_run.sh')
    parser.add_argument('--redir', action='store_true')
    parser.add_argument('--gen_type', type=str, default='baseline')
    args = parser.parse_args()
    if args.gen_type == 'baseline':
        cmds = gen_baseline_cmds(args, args.dataset)
    elif args.gen_type == 'opt':
        cmds = gen_searchM_cmds(args, args.dataset)

    with open(args.fp, 'w') as f:
        for cmd in cmds:
            f.write('{}\n'.format(cmd))
    


