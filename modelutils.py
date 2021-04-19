from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy.sparse as sp
import seaborn as sns
from pygcn.utils import normalize,sparse_mx_to_torch_sparse_tensor,load_data,accuracy
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from pygcn.models import GCN


class AdamOptim():
    def __init__(self, eta=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.m_dw, self.v_dw = 0, 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta = eta
    def update(self, t, w, dw):
        self.m_dw = self.beta1*self.m_dw + (1-self.beta1)*dw
        self.v_dw = self.beta2*self.v_dw + (1-self.beta2)*(dw**2)
        m_dw_corr = self.m_dw/(1-self.beta1**t)
        v_dw_corr = self.v_dw/(1-self.beta2**t)
        w = w - self.eta*(m_dw_corr/(np.sqrt(v_dw_corr)+self.epsilon))
        return w

class Agent:
    def __init__(self,n,m,i,j,eigvecs):

        phi = eigvecs[m*i+j,:]
        l = len(phi)
        f = 4*n*m
        north,east,south,west = np.zeros((1,f)),np.zeros((1,f)),np.zeros((1,f)),np.zeros((1,f))
        north[0,0:l],east[0,l:2*l],south[0,2*l:3*l],west[0,3*l:4*l] = phi,phi,phi,phi
        self.x = [north,east,south,west]
        
class Params:
    def __init__(self,n,m,nt,mt,gamma,qstep,pstep,alpha,noepi,verbose =  False):
               
        self.gamma = gamma 
        self.qstep = qstep 
        self.pstep = pstep 
        self.alpha = alpha
        self.n = n
        self.m = m
        self.nt = nt
        self.mt = mt
        self.w = np.random.randn(4*self.n*self.m,1)*0
        self.theta = np.random.randn(4*self.n*self.m,1)*0
        self.noepi = noepi
        self.verbose = verbose

def ActionSelector(bot,i,j,theta,k):
    
    x = np.random.uniform(0,1)
    if x > 1/500:
        policy = PolicyUpdate(bot,i,j,theta)
    else:
        policy = [0.25,0.25,0.25,0.25]
        
    action = np.random.choice([0,1,2,3],size = 1,p = policy) 
        
    return int(action)

def StateSelector(n,m,i,j,action):
          
    x, y = i, j 
    if action == 0:
        x = i - 1 
    elif action == 1:
        y = j + 1 
    elif action == 2:
        x = i + 1 
    else:
        y = j - 1 
                  
    if x > n-1:
        x = n-1
    elif x < 0:
        x = 0 
   
    if y > m-1:
        y = m-1 
    elif y < 0:
        y = 0 
          
    return int(x),int(y)        
  
def PolicyUpdate(bot,i,j,theta):
    
    value = list(map(lambda x: np.dot(x,theta)[0][0], bot[i,j].x)) 
    value = value - max(value) 
    e = list(map(lambda x: np.exp(x), value)) 
    return np.array([e[0],e[1],e[2],e[3]])*(1/np.sum(e)) 

      
def Qvalue(n,m,bot,weigth):
    for i in range(n):
        for j in range(m):
            values = list(map(lambda x: np.dot(x,weigth), bot[i,j].x))
            print(np.argmax(values),end=" ")
        print()
    print()
        
        
def PlotAnalysis(interval,reg,val,regcn,valgcn,gcn_phi):

#   gcn_phi potential function learnt from the GCN 
    fig = plt.figure()
    fig.suptitle('GCN_Phi potential function', fontsize=20)
    plt.imshow(gcn_phi, cmap='hot', interpolation='nearest')
    sns.heatmap(gcn_phi)
    plt.show()
    
#   Regret in terms of cumulative steps  
    fig = plt.figure()
    fig.suptitle('Cumulative Steps - Regret', fontsize=20)
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('Regret', fontsize=16)
    plt.plot(range(interval),reg[:interval], 'r', label='AC')
    plt.legend()
    plt.plot(range(interval),regcn[:interval],'b', label='AC + Phi')
    plt.legend()
    plt.show()
    
#   Steps taken from start to goal state vs Epoch
    fig = plt.figure()
    fig.suptitle('Steps taken Start to Goal vs Epochs', fontsize=20)
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('Steps', fontsize=16)
    plt.plot(range(interval),val[:interval], 'r', label='AC')
    plt.legend()
    plt.plot(range(interval),valgcn[:interval],'b', label='AC + Phi')
    plt.legend()
    plt.show()
        
def ACPhi(param,reward,args):
    
    n = param.n
    m = param.m
    nt = param.nt
    mt = param.mt
    noepi = param.noepi
    gamma = param.gamma
    alpha = param.alpha
    qstep = param.qstep
    pstep = param.pstep
    w = param.w
    theta = param.theta
    verbose = param.verbose
    N = 10
    
    A = np.zeros((n*m,n*m))
    D = np.eye(n*m)
    gcn_phi = np.zeros((n,m))
    
    adam1 = AdamOptim(eta=pstep)
    adam2 = AdamOptim(eta=qstep)
        
    valgcn = []
    regcn = []
    maxigcn = []

    features,bot = GraphConfig(n,m,A,D)
    
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    gcn_model = GCN(nfeat=n*m, nhid=args.hidden)
    gcn_model.to(device)
    optimizer = optim.Adam(gcn_model.parameters(),lr=args.lr, weight_decay=args.weight_decay)
    
    for iterations in range(noepi):
        i = 0
        j = 0
        
        minsteps = n-1-i + m-1-j
        rew = 0
        epilen = 0
        
        if noepi % N == 0:
            A,D,idx_train,labels = GraphCons(n,m,nt,mt,A,D,(n*m)**2)
            features,bot = GraphConfig(n,m,A,D)
            update_graph(n,m,args,gcn_model,optimizer,features,labels,idx_train,A,D)
            gcn_features, gcn_adj, _, _, _, _ = GCN_inputs(features,labels,idx_train,A,D)
            gcn_phi = gcn_model(gcn_features,gcn_adj).clone().cpu().detach().numpy()[:,1].reshape(n,m)
            gcn_phi = np.exp(gcn_phi)

    
        a = ActionSelector(bot,i,j,theta,(iterations+1))
        
        
        
        while (i!=nt-1 or j!=mt-1):
            
            i_t,j_t = StateSelector(n,m,i,j,a) #n,m
            a_t = ActionSelector(bot,i_t,j_t,theta,(iterations+1))
            q_t = np.dot(bot[i,j].x[a],w) 
            q_t1 = np.dot(bot[i_t,j_t].x[a_t],w)
            
            
            delta2 = reward[i,j,a] + gamma*(q_t1 + gcn_phi[i_t,j_t]) - (q_t + gcn_phi[i,j])
            delta1 = reward[i,j,a] + gamma*q_t1 - q_t 
            
            rew += reward[i,j,a]
            epilen += 1  
            
            north,east,south,west = bot[i,j].x
            n_p,e_p,s_p,w_p = PolicyUpdate(bot,i,j,theta)
            dlogtheta = bot[i,j].x[a].T - ((north.T)*n_p + (east.T)*e_p + (south.T)*s_p + (west.T)*w_p)
            theta = adam1.update(iterations+1,theta,-q_t*dlogtheta)
            w = adam2.update(iterations+1,w,-(bot[i,j].x[a].T)*(alpha*delta1 + (1-alpha)*delta2))
            
            a = a_t
            i = i_t
            j = j_t
    
        if iterations%100 == 0 and verbose == True:
            Qvalue(n,m,bot,w) # n,m
            Qvalue(n,m,bot,theta)
            print(rew)
            print(iterations)

        print(iterations,D-A,'\n\n')
    
        regcn.append(- sum(maxigcn) + sum(valgcn))
        valgcn.append(epilen) 
        maxigcn.append(minsteps)
        
    return regcn,valgcn,gcn_phi

def GraphConfig(n,m,A,D):

    if np.sum(A) == 0:
        features  = np.random.uniform(size = (n*m,n*m))
    else:
        D_hat = la.fractional_matrix_power(D, -0.5)
        L_norm = np.identity(n*m) - np.dot(D_hat, A).dot(D_hat)
        _, features = la.eig(L_norm)
    
    bot = Begin(n,m,features.real) 
    
    return features,bot

def GraphCons(n,m,nt,mt,A,D,sam_len):
    
    i =  0
    j =  0
    states = [] 
    policy = [0.25,0.25,0.25,0.25] 
    a = np.random.choice([0,1,2,3],size = 1,p = policy) 
    states.append(0) 
    epi_len = 0 
    labels = np.zeros((n*m))
    states.append(0)
    
    for episode in range(sam_len):
        
        i_t,j_t = StateSelector(n,m,i,j,a)
        a_t = np.random.choice([0,1,2,3],size = 1,p = policy)
        A[m*i+j,m*i_t+j_t] = 1
        if (i_t == nt-1 and j_t == mt-1):
            labels[m*i+j] = 1  
        states[1] = m*i+j 
        i = i_t
        j = j_t
        a = a_t

    
    D = np.diag(A.sum(axis=1))

    return A,D,states,labels 

def GCN_inputs(features,labels,states,A,D):
    
    gcn_features = normalize(sp.csr_matrix(features))
    gcn_features = torch.FloatTensor(np.array(gcn_features.todense()))
    
    adj = sp.coo_matrix(A) 
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    deg = np.diag(adj.toarray().sum(axis=1))
    laplacian = torch.from_numpy((deg - adj.toarray()).astype(np.float32))
    adj = normalize(sp.csr_matrix(adj) + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    
    labels = torch.LongTensor(labels)
    idx_train = torch.LongTensor(states)
    
    return gcn_features, adj, deg, laplacian, labels, idx_train


def Begin(n,m,eigvecs):
    bot = []
    for i in range(n):
        temp = []
        for j in range(m):
            temp.append(Agent(n,m,i,j,eigvecs)) 
        bot.append(temp)  
    return np.array(bot) 

def update_graph(n,m,args,model,optimizer,features,labels,states,adj,degree):

    features, adj, deg, laplacian, labels, idx_train = GCN_inputs(features,labels,states,adj,degree)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda and torch.cuda.is_available():
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        laplacian =laplacian.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()

    t_total = time.time()
    for epoch in range(args.gcn_epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        soft_out= torch.unsqueeze(torch.nn.functional.softmax(output,dim=1)[:,1],1)
        loss_reg  = torch.mm(torch.mm(torch.transpose(soft_out, 0, 1),laplacian),soft_out)
        print('Epoch: {:04d}'.format(epoch+1),
        'loss_train: {:.4f}'.format(loss_train.item()),
        'time: {:.4f}s'.format(time.time() - t))
        loss_train +=  args.gcn_lambda * loss_reg.squeeze()
        loss_train.backward()
        optimizer.step()