from modelutils import *
import torch
import torch.nn.functional as F
import torch.optim as optim
from pygcn.models import GCN
from pygcn.utils import normalize,sparse_mx_to_torch_sparse_tensor,load_data,accuracy
from args import get_args
from pygcn.GCN_train import *

args = get_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    
def main():

    n = args.n
    m = args.m
    nt = args.nt
    mt = args.mt

    reward = np.ones((n,m,4))*-0.1
    reward[nt-2,mt-1,2],reward[nt-1,mt-2,1] = 0,0 

    # torch.set_num_threads(1)
    # device = torch.device("cuda:0" if args.cuda else "cpu")
    # gcn_model = GCN(nfeat=n*m, nhid=args.hidden)
    # gcn_model.to(device)
    # optimizer = optim.Adam(gcn_model.parameters(),lr=args.lr, weight_decay=args.weight_decay)
    # adj,features,_,_ = shortest_dist(n,m,[n,m])

    # features = normalize(sp.csr_matrix(features))
    # features = torch.FloatTensor(np.array(features.todense()))

    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # adj = normalize(sp.csr_matrix(adj) + sp.eye(adj.shape[0]))
    # adj = sparse_mx_to_torch_sparse_tensor(adj)


    # for episodes in range(args.gcn_epi):
    #     update_graph(n,m,args,gcn_model,optimizer)
    #     print("{} episode done :".format(episodes+1))
    
    # if args.cuda and torch.cuda.is_available():
    #     features = features.cuda()
    #     adj = adj.cuda()

    # output = gcn_model(features,adj).cpu()
    # gcn_phi = torch.exp(output).detach().numpy()
    # gcn_phi = gcn_phi[:,1].reshape(n,m)

    
    param1 = Params(n,m,nt,mt,gamma = args.gamma,qstep = args.qstep,pstep = args.pstep,alpha = 0,noepi = args.noepi)
    param2 = Params(n,m,nt,mt,gamma = args.gamma,qstep = args.qstep,pstep = args.pstep,alpha = 1,noepi = args.noepi)
    
    regcn,valgcn = ACPhi(param1,reward,args)
    reg,val = ACPhi(param2,reward,args)

    PlotAnalysis(param1.noepi,reg,val,regcn,valgcn,gcn_phi)

if __name__ == "__main__":
    main()
    