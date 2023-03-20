import ast
import argparse
from itertools import product
import numpy as np
import optuna
import os
import re

# please keep this variable same as that in batch_run.sh
FIX_ARGS = "--coef_upd pcd --emb mlp"

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
    
def args2str(args):
    ret = ''
    for k,v in args.items():
        ret += ' --{}'.format(k)+' '+str(v)
    return ret

def parse_from_files(args):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    files = os.listdir(root_dir)
    prefix = 'search' if not args.psearch else 'psearch'
    files = list(filter(lambda x: x.startswith(prefix) and x.endswith('.out'), files))
    print(files)
    opt_by_ds = {}
    cmds = []
    for fp in files:
        tfp = fp[:-4]
        print(tfp)

        arr = tfp.split('_')
        if len(arr)<4:
            continue
        model = arr[1]
        ds = arr[3]
        # if specific model or dataset
        if args.model is not None:
            if model != args.model:
                continue
        if args.dataset is not None:
            if ds != args.dataset:
                continue
        best_param, best_val = get_best_param_res(os.path.join(root_dir, fp))

        if ds not in opt_by_ds.keys():
            opt_by_ds[ds] = [{'model':model, 'param':best_param, 'val':best_val}]
        else:
            opt_by_ds[ds].append({'model':model, 'param':best_param, 'val':best_val})

    for ds,arr in opt_by_ds.items():
        for rec in arr:
            model = rec['model']
            best_param = rec['param']
            best_val = rec['val']
            if best_param is None:
                print('Running: {} {}'.format(ds, model))
                continue
            else: print('Found {} {}: val acc {}'.format(ds, model, best_val), best_param)
            best_param_str = args2str(best_param)
            cmd = f'python train_model.py --model {model} --dataset {ds} {best_param_str}'
            if args.gpu_id >= 0:
                cmd  = cmd + ' --gpu_id {}'.format(args.gpu_id)
            if FIX_ARGS:
                cmd = cmd + ' ' + FIX_ARGS
            if args.redir:
                fp_name = 'test'+'_{}_{}'.format(ds, model)
                cmd  = cmd + ' > {}.out 2> {}.err'.format(fp_name, fp_name)
            cmds.append(cmd)

    with open(args.fp, 'w') as f:
        for cmd in cmds:
            f.write('{}\n'.format(cmd))

def parse_from_optuna(args):
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(repo_dir, 'param_search')
    files = os.listdir(root_dir)
    opt_by_ds = {}
    cmds = []
    for fp in files:
        if not fp.endswith('.sqlite'):
            continue
        tfp = fp[:-7]
        arr = tfp.split('_')
        model = arr[1]
        ds = arr[3]
        print(tfp)

        # if specific model or dataset
        if args.model is not None:
            if model != args.model:
                continue
        if args.dataset is not None:
            if ds != args.dataset:
                continue
        study = optuna.load_study(study_name='opt_'+model+'_'+ds,storage='sqlite:///'+os.path.join(root_dir,fp))
        if ds not in opt_by_ds.keys():
            opt_by_ds[ds] = [{'model':model, 'opt':study}]
        else:
            opt_by_ds[ds].append({'model':model, 'opt':study})

    for ds,arr in opt_by_ds.items():
        for rec in arr:
            model = rec['model']
            opt = rec['opt']
            print('Found {} {}: val acc {}'.format(ds, model, opt.best_value), opt.best_params)
            best_params = opt.best_params
            best_param_str = args2str(best_params)
            cmd = f'python train_model.py --model {model} --dataset {ds} {best_param_str}'
            if args.gpu_id >= 0:
                cmd  = cmd + ' --gpu_id {}'.format(args.gpu_id)
            if args.redir:
                fp_name = 'test_{}_{}'.format(ds, model)
                cmd  = cmd + ' > {}.out 2> {}.err'.format(fp_name, fp_name)
            cmds.append(cmd)

    with open(args.fp, 'w') as f:
        for cmd in cmds:
            f.write('{}\n'.format(cmd))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--gpu_id', type=int, default=-1)
    parser.add_argument('--fp', type=str, default='./test_run.sh')
    parser.add_argument('--redir', action='store_true')
    parser.add_argument('--psearch', action='store_true')
    args = parser.parse_args()

    parse_from_files(args)
