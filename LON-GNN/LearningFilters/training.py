from utils import filtering,visualize,TwoDGrid
import numpy as np
from functools import partial
import os 
import argparse
import torch
from models import ChebNet,BernNet,GcnNet,GatNet,ARMANet,GPRNet
from orthnorm import JacobiSGNN, StdJacobiSGNN, JacobiSGNNS, StdJacobiSGNNS
from sklearn.metrics import r2_score

from search_params import SEARCH_PARAMS

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--early_stopping', type=int,default=200)
parser.add_argument('--lr',type=float,default=0.01)
parser.add_argument('--filter_type',type=str,choices=['low','high','band','rejection','comb','low_band'],default='band')
parser.add_argument('--net',type=str,choices=['ChebNet','BernNet','GcnNet','GatNet','ARMANet','GPRNet', 'JacobiSGNN', 'StdJacobiSGNN', 'JacobiSGNNS', 'StdJacobiSGNNS'],default='BernNet')
parser.add_argument('--img_num',type=int,default=50)
parser.add_argument('--gpu_id', type=int, default=0)

# args for jacobiSGNN, etc.
parser.add_argument('--param_search', action='store_true')
parser.add_argument('--mulfilter', type=bool, default=False)
parser.add_argument('--coef_upd', type=str, default='none')
parser.add_argument('--jacob_a', type=float, default=-0.75)
parser.add_argument('--jacob_b', type=float, default=0.25)
parser.add_argument('--alpha', type=float, default=2.)
parser.add_argument('--lr_ab', type=float, default=0.001)
parser.add_argument('--wd', type=float, default=0.0001)
parser.add_argument('--wd_ab', type=float, default=0.0)
args = parser.parse_args()
print(args)

if os.path.exists('y_'+args.filter_type+'.npy'):
	y=np.load('y_'+args.filter_type+'.npy')
else:
	y=filtering(args.filter_type)
y=torch.Tensor(y)
dataset = TwoDGrid(root='data/2Dgrid', pre_transform=None)
data=dataset[0]

device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')

y=y.to(device)
data=data.to(device)

def train(img_idx,model,optimizer):
    model.train()
    optimizer.zero_grad()
    pre=model(data)
    loss= torch.square(data.m*(pre- y[:,img_idx:img_idx+1])).sum()        
    loss.backward()
    optimizer.step()
    a=pre[data.m==1]    
    b=y[:,img_idx:img_idx+1] 
    b=b[data.m==1]
    r2=r2_score(b.cpu().detach().numpy(),a.cpu().detach().numpy())

    return loss.item(),r2

def run(args, see=True):
	results=[]
	for img_idx in range(args.img_num):
		data.x_tmp=data.x[:,img_idx:img_idx+1]

		if args.net=='ChebNet':
			model=ChebNet().to(device)
		elif args.net=='BernNet':
			model=BernNet().to(device)
		elif args.net=='GcnNet':
			model=GcnNet().to(device)
		elif args.net=='GatNet':
			model=GatNet().to(device)
		elif args.net=='ARMANet':
			model=ARMANet().to(device)
		elif args.net=='GPRNet':
			model=GPRNet().to(device)
		elif args.net=='JacobiSGNN':
			model=JacobiSGNN(args).to(device)
		elif args.net == 'JacobiSGNNS':
			model = JacobiSGNNS(args).to(device)
		elif args.net=='StdJacobiSGNN':
			model=StdJacobiSGNN(args).to(device)
		elif args.net=='StdJacobiSGNNS':
			model=StdJacobiSGNNS(args).to(device)
		

		if args.net not in ['JacobiSGNN', 'JacobiSGNNS', 'StdJacobiSGNN', 'StdJacobiSGNNS']:
			optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
		else:
			optimizer = torch.optim.Adam([
				{
					'params': model.mf_weights,
					'weight_decay': args.wd,
					'lr': args.lr
				},
				{
					'params': [model.a, model.b],
					'weight_decay': args.wd_ab,
					'lr': args.lr_ab
				}
				])

		best_r2=0
		min_loss=float('inf')
		cnt=0
		re_epoch=args.epochs

		for epoch in range(args.epochs):
			loss,r2=train(img_idx,model,optimizer)
			if(min_loss>loss):
				min_loss=loss
				best_r2=r2
				cnt=0
			else:
				cnt+=1
			if(cnt>args.early_stopping):
				re_epoch=epoch+1
				break

		results.append([min_loss,best_r2])
		if True:
			print(model.a.detach(), model.b.detach())
			print(f'img: {img_idx+1} \t loss= {min_loss:.4f} \t r2= {best_r2:.4f} \t epoch: {re_epoch}')
		
	loss_mean, r2_mean= np.mean(results, axis=0)
	if see:
		print(f'mean_loss={loss_mean:.6f} mean_r2_acc={r2_mean:.4f}')
	return loss_mean, r2_mean

def param_search(trial, args):
	params = SEARCH_PARAMS
	for key in params.keys():
		setattr(args, key, trial.suggest_categorical(key, params[key]))
	args.jacob_a = trial.suggest_float('jacob_a', -0.995, 0.05, step=0.005)
	if args.net in ['StdJacobiSGNN', 'StdJacobiSGNNS']:
		args.jacob_b = trial.suggest_float('jacob_b', max(-0.2, -(1.0+args.jacob_a)+0.00001), 3.5, step=0.05)
	else:
		args.jacob_b = trial.suggest_float('jacob_b', -0.2, 3.5, step=0.05)
	loss, r2 = run(args, see=False)
	trial.set_user_attr('r2', r2)
	return loss

if __name__ == '__main__':
	if args.param_search:
		import optuna
		study = optuna.create_study(direction='minimize', study_name = 'opt_'+args.net+'_'+args.filter_type, load_if_exists=False, storage=None)
		study.optimize(partial(param_search, args=args), n_trials=100)
		print(study.best_params)
		best_value = study.best_value
		best_trial = study.best_trial
		best_r2 = best_trial.user_attrs['r2']
		print(f'mean_loss={best_value:.4f} mean_r2_acc={best_r2:.4f}')
	else:
		run(args, True)


