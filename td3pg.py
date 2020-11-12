import torch as T
import numpy as np  
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import os
from replay_buffer import replay_buffer
from nets import critic,actor



# Implementation of Addressing Function Approximation Error in Actor-Critic Methods
# :https://arxiv.org/abs/1802.09477


class agent():

    def __init__(self,alpha,beta,inp_dims,tau,env,gamma=0.99,update_actor_int = 2,warmup = 1000,
                 n_actions= 2,max_size = 1000000,l1_size = 400,l2_size = 300,batch_size =100,noise = 0.1):
        

        self.tau =tau
        self.gamma = gamma
        self.min_act  = env.action_space.low
        self.max_act = env.action_space.high
        self.mem = replay_buffer(max_size,inp_dims,n_actions)

        self.batch_size = batch_size
        self.lrn_step_count = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.time_step=0
        


        self.update_act_iter = update_actor_int

        self.actor = actor(alpha,inp_dims,l1_size,l2_size,n_actions= n_actions,name="Actor")



        self.critic1 = critic(beta,inp_dims,l1_size,l2_size,n_actions= n_actions,name="Critic1")
        self.critic2 = critic(beta,inp_dims,l1_size,l2_size,n_actions= n_actions,name="Critic2")

        self.actor_target =actor(alpha,inp_dims,l1_size,l2_size,n_actions= n_actions,name="Actor_target")
        self.critic_target1 =  critic(beta,inp_dims,l1_size,l2_size,n_actions= n_actions,name="Critic_target1")
        self.critic_target2  =  critic(beta,inp_dims,l1_size,l2_size,n_actions= n_actions,name="Critic_target2")


        self.noise = noise
        self.update_net_params(tau=1)


    def action_choose(self,obsv):
        if self.time_step < self.warmup:

            mu=T.tensor(np.random.normal(scale=self.noise,size = (self.n_actions,))).to(self.actor.device)
        
        else:
            state = T.tensor(obsv,dtype=T.float).to(self.actor.device)
            mu = self.actor.forward(state).to(self.actor.device)
        mu_prime = mu+T.tensor(np.random.normal(scale=self.noise),dtype=T.float).to(self.actor.device)
        mu_prime = T.clamp(mu_prime,self.min_act[0],self.max_act[0])
        self.time_step+=1

        return mu_prime.cpu().detach().numpy()

    def rem_transition(self,state,action,reward,nw_state,done):
        self.mem.transition_store(state,action,reward,nw_state,done)

    def learning(self):

        if self.mem.count_mem < self.batch_size:
            return

        state,action,reward,new_state,done = self.mem.sample_buffer(self.batch_size)
        reward=T.tensor(reward,dtype=T.float).to(self.critic1.device)
        done = T.tensor(done).to(self.critic1.device)
        nw_state = T.tensor (new_state,dtype=T.float).to(self.critic1.device)
        state = T.tensor (state,dtype=T.float).to(self.critic1.device)
        action= T.tensor (action,dtype=T.float).to(self.critic1.device)


        target_act = self.actor_target.forward(nw_state)
        target_act = target_act + T.clamp(T.tensor(np.random.normal(scale = 0.2)),-0.5,0.5)

        target_act = T.clamp(target_act,self.min_act[0],self.max_act[0])
        q1_ = self.critic_target1.forward(nw_state,target_act)
        q2_ = self.critic_target2.forward(nw_state,target_act)

        q1 = self.critic1.forward(state,action)
        q2 = self.critic2.forward(state,action)


        q1_[done] = 0.0
        q2_[done] = 0.0
        q1_ = q1_.view(-1)
        q2_ =q2_.view(-1)


        critic_val =T.min(q1_,q2_)


        target =reward + self.gamma*critic_val
        target = target.view(self.batch_size,1)
        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()


        loss_q1 = F.mse_loss(target,q1)
        loss_q2 = F.mse_loss(target,q2)

        critic_loss = loss_q1+loss_q2
        critic_loss.backward()
        self.critic1.optimizer.step()
        self.critic2.optimizer.step()

        self.lrn_step_count +=1


        if self.lrn_step_count % self.update_act_iter !=0:
            return

        self.actor.optimizer.zero_grad()
        act1_q1_loss = self.critic1.forward(state,self.actor.forward(state))
        act_loss = -T.mean(act1_q1_loss)
        act_loss.backward()
        self.actor.optimizer.step()
        self.update_net_params()


    def update_net_params(self,tau=None):
        if tau is None:
            tau =self.tau

        actor_params=self.actor.named_parameters()
        critic1_params = self.critic1.named_parameters()
        critic2_params = self.critic2.named_parameters()
        target_actor_params = self.actor_target.named_parameters()
        target_critic1_params =self.critic_target1.named_parameters()
        target_critic2_params = self.critic_target2.named_parameters()

        actor_ =dict(actor_params)
        critic1_ =dict(critic1_params)
        critic2_ =dict(critic2_params)
        actor_target_ =dict(target_actor_params)
        target_critic1_ = dict(target_critic1_params)
        target_critic2_ = dict(target_critic2_params)


        for nm in critic1_ : 
            critic1_[nm]  = tau*critic1_[nm].clone() + (1-tau)*target_critic1_[nm].clone()

        for nm in critic2_ : 
            critic2_[nm]  = tau*critic2_[nm].clone() + (1-tau)*target_critic2_[nm].clone()


        for nm in actor_:
            actor_[nm] = tau*actor_[nm].clone() + (1-tau)*actor_target_[nm].clone()

        self.critic_target1.load_state_dict(critic1_)
        self.critic_target2.load_state_dict(critic2_)
        self.actor_target.load_state_dict(actor_)

    def model_load(self):
        self.actor.checkpoint_load()
        self.actor_target.checkpoint_load()
        self.critic1.checkpoint_load()
        self.critic2.checkpoint_load()
        self.critic_target1.checkpoint_load()
        self.critic_target2.checkpoint_load()

    def model_save(self):
        self.actor.checkpoint_save
        self.actor_target.checkpoint_save()
        self.critic1.checkpoint_save()
        self.critic2.checkpoint_save()
        self.critic_target1.checkpoint_save()
        self.critic_target2.checkpoint_save()



        
        
        

