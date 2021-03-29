import argparse

import torch

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--gamma', type=int, default=1,
                    help='Discounting factor')
    parser.add_argument('--alpha', type=float, default=0,
                        help='alpha for Q_combined')
    parser.add_argument('--qstep', type=float, default=0.01,
                    help='step size for critic')
    parser.add_argument('--pstep', type=float, default=0.005,
                    help='step size of actor')
    parser.add_argument('--n', type=int, default=15,
                    help='number of rows in grid')
    parser.add_argument('--m', type=int, default=15,
                    help='number of columns in grid')
    parser.add_argument('--nt', type=int, default=15,
                    help='goalstate row')
    parser.add_argument('--mt', type=int, default=15,
                    help='goalstate column')
    parser.add_argument('--noepi', type=int, default=2000,
                    help='number of episodes')
    parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
    parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0e-4,
                    help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
    parser.add_argument('--gcn_epochs', type=int, default= 1,
                    help='Number of epochs to train.')  
    parser.add_argument('--gcn_epi', type=int, default= 40,
                    help='Number of episodes to train gcn')  
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--gcn_lambda', type=float, default=0.1,
                        help='Mixing coefficient between GCN losses.')

    parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args