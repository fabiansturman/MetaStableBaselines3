#THIS SCRIPT IS ALL ABOUT TRYING OUT MY SMC FOR THEOREM 1 AND SEEING WHAT HAPPENS!

#print("THIS SCRIPT USES *WINDOWS* FILEPATHS. CHANGE FOR LINUX!! (inc the additional 'proj_code' at start for some reason)")

####################################################################################
print("Imports...")
import random

import numpy as np
import torch
from torch import optim
from tqdm import tqdm

import gymnasium as gym

from stable_baselines3 import A2C,PPO


from stable_baselines3.common.logger import configure


import fabian.envs.khazad_dum_gymn 

import matplotlib.pyplot as plt

import pickle


####################################################################################
print("Setting up...")
model_save_path = "proj_code/"+"saved_models/27May_MAML_A2C_KDcont3_ep64" #simulatenously testing my new maml syntax out and doing PPO (if somethings up, then try A2C with my new maml syntax to see if its my PPO that is wrong or the MAML syntax (too?))


model_save_path_rob = "proj_code/"+"saved_models/27May_MAML_A2C_KDcont3_fakeCVaR3"

device='cpu'
adapt_lr =  7e-4
adapt_timesteps = 32*4 #for this enviornment, each episode is exactly 32 timesteps, so multiple of 32 means full number of eps experienced for each task
eval_timesteps = 64 # evaluate one full epsiode at a time
M=1

env = gym.make("KhazadDum-v1",continuous=True, max_speed=0.5, max_episode_steps=64) # can access wrapped env with "env.unwrapped" (e.g. to reset task)
env.unwrapped.exp_bonus = 1; env.unwrapped.bridge_bonus_factor = 2 #this should incentivise getting to the target asap, and incentivise going onto the bridge


meta_agent = A2C("MlpPolicy", env, verbose=0, learning_rate=adapt_lr, device=device,
                 meta_learning=True, M=M, adapt_timesteps=adapt_timesteps, eval_timesteps=eval_timesteps)
meta_agent.policy.load_state_dict(torch.load(f"{model_save_path}/final", weights_only=True))

meta_agent_rob = A2C("MlpPolicy", env, verbose=0, learning_rate=adapt_lr, device=device,
                 meta_learning=True, M=M, adapt_timesteps=adapt_timesteps, eval_timesteps=eval_timesteps)
meta_agent_rob.policy.load_state_dict(torch.load(f"{model_save_path_rob}/final", weights_only=True))
'''
####################################################################################
print("Return generation for SMC: generating totally i.i.d returns")

tasks=100_000

returns = meta_agent.sample_returns(tasks=tasks, repeats_per_task=1)
return_list = list(returns.values())

try:
    dbfile = open(f"{model_save_path}/iidReturnList.pickle", 'rb')
    return_list=pickle.load(dbfile)+return_list
except:
    print("making the file")


dbfile = open(f"{model_save_path}/iidReturnList.pickle", 'wb')
pickle.dump(return_list, dbfile)



####################################################################################
print("Return generation for SMC: generating totally batches returns from tasks ")

tasks=200

returns = meta_agent.sample_returns(tasks=tasks, repeats_per_task=50)

try:
    dbfile = open(f"{model_save_path}/TaskwiseReturns.pickle", 'rb')
    returns=pickle.load(dbfile) + returns
except:
    print("making the file")

dbfile = open(f"{model_save_path}/TaskwiseReturns.pickle", 'wb')
pickle.dump(returns, dbfile)



#############
print(len(return_list))
print(len(returns.keys()))

###########################
'''
dim = 4
fig, axs = plt.subplots(dim, dim, figsize=(10, 8))

for t in tqdm(range(dim*dim)):
    x = t//dim
    y = t%dim
    if y ==0:
        if x==0:
            env.unwrapped.task=0.002
        if x == 1:
            env.unwrapped.task=0.126
        if x == 2:
            env.unwrapped.task=0.452
        if x == 3:
            env.unwrapped.task=1.354
        axs[y,x].set_title(f"Task {env.unwrapped.task}")
    agent = meta_agent if y in [0,1] else  meta_agent_rob


    #Perform few shot adaption to environment
    _,_,adapted_policy = agent.meta_adapt(task = env.unwrapped.get_task(), M=1)
    adapted_policy = adapted_policy[0] # we set M=1 above so only 1 policy adapted here

    #Test against a new trajectory from that state (else we are showing something it trained to and before the final training step)
    agent.evaluate_policy(total_timesteps=32, policy=adapted_policy)

    #Plot this run
    axs[y,x] = env.unwrapped.show_state(axs[y,x],show_task=False, text_coords=(2,1))    
    axs[y,x].set_axis_off()

plt.tight_layout()
plt.show()