import torch as T
import numpy as np  
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import os



class critic(nn.Module):
    
    def __init__(self,beta,inp_dims,fcl1_dims,fcl2_dims,n_actions,name,checkpoint_dir = "temp/td3pg"):


        super(critic,self).__init__()

        self.inp_dims = inp_dims
        self.fcl1_dims = fcl1_dims
        self.fcl2_dims = fcl2_dims
        self.n_actions = n_actions
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,name+"_td3pg")
        self.name = name
        

        self.fcl1 =nn.Linear(self.inp_dims[0]  + n_actions,self.fcl1_dims)
        self.fcl2 = nn.Linear(self.fcl1_dims,self.fcl2_dims)
        self.q1 = nn.Linear(self.fcl2_dims,1)
        self.optimizer = optim.Adam(self.parameters(),lr=beta)

        self.device  = T.device("cuda:0" if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self,state,action):
        q1_act_val = self.fcl1(T.cat([state,action],dim=1))
        q1_act_val=F.relu(q1_act_val)
        q1_act_val = self.fcl2(q1_act_val)
        q1_act_val = F.relu(q1_act_val)

        q1 = self.q1(q1_act_val)


        return q1


    def checkpoint_save(self):
        print("................Saving the checkpoint............")
        T.save(self.state_dict(),self.checkpoint_file)

    def checkpoint_load(self):
        print("................Loading the checkpoint............")
        self.load_state_dict(T.load(self.checkpoint_file))



class actor(nn.Module):
    
    def __init__(self,alpha,inp_dims,fcl1_dims,fcl2_dims,n_actions,name,checkpoint_dir = "temp/td3pg"):


        super(actor,self).__init__()

        self.inp_dims = inp_dims
        self.fcl1_dims = fcl1_dims
        self.fcl2_dims = fcl2_dims
        self.n_actions = n_actions
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,name+"_td3")
        self.name = name
        

        self.fcl1 =nn.Linear(*self.inp_dims  ,self.fcl1_dims)
        self.fcl2 = nn.Linear(self.fcl1_dims,self.fcl2_dims)
        self.mu = nn.Linear(self.fcl2_dims,self.n_actions)
        self.optimizer = optim.Adam(self.parameters(),lr=alpha)

        self.device  = T.device("cuda:0" if T.cuda.is_available() else 'cpu')

        self.to(self.device)


    def forward(self,state):
        prb=self.fcl1(state)
        prb= F.relu(prb)
        prb=self.fcl2(prb)
        prb= F.relu(prb)
        prb=self.mu(prb)
        mu=T.tanh(prb)


        return mu


    def checkpoint_save(self):
        print("................Saving the checkpoint............")
        T.save(self.state_dict(),self.checkpoint_file)

    def checkpoint_load(self):
        print("................Loading the checkpoint............")
        self.load_state_dict(T.load(self.checkpoint_file))