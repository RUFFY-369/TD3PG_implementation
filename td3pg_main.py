import numpy as np
import gym
from gym import wrappers
from td3pg import agent
from utils import plot_learning_curve



# Implementation of Addressing Function Approximation Error in Actor-Critic Methods
# :https://arxiv.org/abs/1802.09477


if __name__ =="__main__":

    
    
    #Uncomment the env_name for experiments on different environments
    #env_name="InvertedPendulum-v1"
    env_name = "HalfCheetah-v1"
    #env_name= "Ant-v1"
    #env_name="LunarLanderContinuous-v2"
    env= gym.make(env_name)
    agent1 =agent(alpha=0.001,beta=0.001,inp_dims=env.observation_space.shape,tau=0.005,env = env,batch_size=100,l1_size=400,l2_size=300,
                  n_actions=env.action_space.shape[0])

    
    #keeping episode_id :True will record for all the episodes which would consume a lot of memory in case of long training
    #so recording every 25th episode to keep track of the training progress
    env = wrappers.Monitor(env, 'temp/video', video_callable=lambda episode_id: episode_id%25==0 , force=True)
    
    
    n_game = 1000
    fig_file ='plots/'+env_name
    best_score = env.reward_range[0]
    score_hist= []
    
    env.render( "rgb_array")

    for i in range(n_game):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent1.action_choose(observation)
            nw_observation, reward, done, _ = env.step(action)
            score += reward
            agent1.rem_transition(observation, action, reward, nw_observation, done)
            agent1.learning()
            observation = nw_observation

        score_hist.append(score)
        avg_score = np.mean(score_hist[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent1.model_save()

        print('episode :', i, 'score %.1f :' % score, 'avg score %.1f :' % avg_score)


    


    
    x = [i+1 for i in range(n_game)]
    plot_learning_curve(x, score_hist, fig_file,n_game)