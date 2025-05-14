#Perform imports
import random

import numpy as np
import torch
from torch import optim
from tqdm import tqdm

import gymnasium as gym


from stable_baselines3 import A2C

from stable_baselines3.common.logger import configure


import matplotlib.pyplot as plt
device = 'cpu'# 'cuda' #  #doing cpu as A2C with MlpPolicy (rather than CNNpolicy) in stablebaseline is faster on CPU, and the meta gradinet beign faster on GPU (even if it is) is not *that* much faster - it is about two(ish) times slower overall based on one run each with two meta iterations, so better on cpu in this case
torch.set_default_device(device)


##################################################################################

model_save_path = "saved_models/metalearningKD_13_expbonus1_bridgebonusfactor2"

##################################################################################
##Set up for meta learning

#Hyperparameters
adapt_lr =  7e-4
meta_lr = 0.0005 
meta_iterations = 500#1250
adapt_timesteps = 32*4 #for this enviornment, each episode is exactly 32 timesteps, so multiple of 32 means full number of eps experienced for each task
tasks_per_loop = 40#60
adapt_visualisations = 15

vis_timesteps = meta_iterations//adapt_visualisations #denominator is number of visualisations we want
if vis_timesteps == 0:
    vis_timesteps = 1 

#Make meta-environment
import fabian.envs.khazad_dum_gymn 
env = gym.make("KhazadDum-v1") # can access wrapped env with "env.unwrapped" (e.g. to reset task)
env.unwrapped.exp_bonus = 1; env.unwrapped.bridge_bonus_factor = 2 #this should incentivise getting to the target asap, and incentivise going onto the bridge

#Make meta-policy and meta-optimiser
meta_agent = A2C("MlpPolicy", env, verbose=0, meta_learning=True, learning_rate=adapt_lr, device=device) #we train the meta_agent to do well at adapting to new envs (i.e. meta learning) in our env distribution
meta_opt = optim.Adam(meta_agent.policy.parameters(), lr=meta_lr)

#Logging variables
meta_losses = []
meta_rets = []
best_meta_ret = None
best_meta_ret_it = -1

##################################################################################

#Load in enviornemnt that performed best against meta validation
loaded_meta_agent = A2C("MlpPolicy", env, verbose=0, meta_learning=True, learning_rate=adapt_lr, device=device)
#loaded_meta_agent.policy.load_state_dict(torch.load(f"{model_save_path}\\best_val_meta_ret", weights_only=True)) 
    #as can be seen in the learning curve above, the model with the best meta-return is very much not the one which actually performs the best (if we see it perform, we see it just geniunely does badly!)
    #TODO: what is going on with that?

#loaded_meta_agent.policy.load_state_dict(torch.load(f"{model_save_path}\\meta_it_{495}", weights_only=True)) 

loaded_meta_agent.policy.load_state_dict(torch.load(f"{model_save_path}/final_model", weights_only=True)) 


    #can also load in intermediate saved envs from training
##################################################################################
#Seeing it manage in a range of environments - this one is taken from loading the best model from a 400 meta-step training, which clearly isnt quite all that 
dim = 3
fig, axs = plt.subplots(dim, dim, figsize=(10, 8))

for t in tqdm(range(dim*dim)):
    #Perform few shot adaption to environment
    env.unwrapped.reset_task() #randomly selects task from environment to reset it to
    loaded_meta_agent.learn(total_timesteps=adapt_timesteps) #adapt the meta agent to this task

    #Test against a new trajectory from that state (else we are showing something it trained to and before the final training step)
    loaded_meta_agent.run_meta_adaption_and_loss(total_timesteps=32)

    #TODO: IMPROTANT TO NOTE - for fair evaluation of a MAML-meta-policy, as this is all about few-shot learning, we need to evaluate its performance to a task after having had a chance to adapt to that task

    #Plot this run
    x = t//dim
    y = t%dim
    axs[x,y] = env.unwrapped.show_state(axs[x,y])    
    axs[x,y].set_axis_off()
    
plt.tight_layout()
plt.savefig(f"{model_save_path}_finalSummaryView")
