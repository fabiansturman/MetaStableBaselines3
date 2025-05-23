#######################################################
#Perform imports
import random

import numpy as np
import torch
from torch import optim
from tqdm import tqdm

import gymnasium as gym


from stable_baselines3 import A2C,PPO

from stable_baselines3.common.logger import configure


import matplotlib.pyplot as plt


device = 'cpu'# 'cuda' #  #doing cpu as A2C with MlpPolicy (rather than CNNpolicy) in stablebaseline is faster on CPU, and the meta gradinet beign faster on GPU (even if it is) is not *that* much faster - it is about two(ish) times slower overall based on one run each with two meta iterations, so better on cpu in this case
torch.set_default_device(device)
#######################################################
model_save_path = "saved_models/19May_TestingPseudoRoML_A2C_1" #simulatenously testing my new maml syntax out and doing PPO (if somethings up, then try A2C with my new maml syntax to see if its my PPO that is wrong or the MAML syntax (too?))

import os
os.mkdir(model_save_path)
#######################################################
print("set seeds")
seed = 0
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

##cuda
if device != 'cpu':
    torch.cuda.manual_seed(seed)

#######################################################
##Set up for meta learning

#Hyperparameters
adapt_lr =  7e-4
meta_lr = 0.0005 
meta_iterations = 500#500#1250
adapt_timesteps = 32*4 #for this enviornment, each episode is exactly 32 timesteps, so multiple of 32 means full number of eps experienced for each task
eval_timesteps = 100
tasks_per_loop = 40#60
adapt_visualisations = 15
M=1

vis_timesteps = meta_iterations//adapt_visualisations #denominator is number of visualisations we want
if vis_timesteps == 0:
    vis_timesteps = 1 

#Make meta-environment
import fabian.envs.khazad_dum_gymn 
env = gym.make("KhazadDum-v1") # can access wrapped env with "env.unwrapped" (e.g. to reset task)
env.unwrapped.exp_bonus = 1; env.unwrapped.bridge_bonus_factor = 2 #this should incentivise getting to the target asap, and incentivise going onto the bridge

#Pseudo-Roml - Add an offset to the sampled noise (for alpha-cvar, it is -mean* ln(alpha) by memoryless property off exponentials)
env.unwrapped.noise_offset = -1*env.unwrapped.task*np.log(0.025) #turned to 0.025 rather than 0.05

#Make meta-policy and meta-optimiser
meta_agent = A2C("MlpPolicy", env, verbose=0, learning_rate=adapt_lr, device=device,
                 meta_learning=True, M=M, adapt_timesteps=adapt_timesteps, eval_timesteps=eval_timesteps) #we train the meta_agent to do well at adapting to new envs (i.e. meta learning) in our env distribution
meta_opt = optim.Adam(meta_agent.policy.parameters(), lr=meta_lr)

#Logging variables
meta_losses = []
meta_rets = []
best_meta_ret = None
best_meta_ret_it = -1


#######################################################
#Outer meta-learning loop
for meta_it in tqdm(range(meta_iterations)):
    meta_loss = 0
    meta_ret = 0
    #Have agent adapt to tasks one by one
    for t in (range(tasks_per_loop)):
        adaptation_loss, a_rets, _ = meta_agent.meta_adapt()
        meta_loss += adaptation_loss
        meta_ret += a_rets

    #Perform gradient update on meta learning parameters
    meta_loss/=tasks_per_loop #normalise, so that learning rate need not depend on tasks/loop
    meta_ret/=tasks_per_loop
    
    meta_opt.zero_grad()
    meta_loss.backward()
    meta_opt.step()

    #Save(/override) the best performing model (w.r.t performance against adaptation task set)
    if best_meta_ret is None or meta_ret>best_meta_ret :
        best_meta_ret = meta_ret
        best_meta_ret_it = meta_it
        torch.save(meta_agent.policy.state_dict(), f"{model_save_path}/best_val_meta_ret") #OK to just save {model.policy} as that is all being meta-optimised
            #TODO: could it be that the highest return happens before the end of training because actually it then optimises loss and that gets us to the target in a way that return doesnt capture? not sure that makes sense so probs not....??? unlike loss being wierd before the critic is trained (I think thta is the source of the wierdness) return should always tell us whats good and what isnt rihgt?

    #Track meta_training curve
    meta_losses.append(meta_loss.detach().item())
    meta_rets.append(meta_ret.item())

  #  #Pause training every 30 meta iterations and wait for input (just for now, so I can safely pause stuff!)
   # if meta_it % 30 == 0:
    #    input(f"Paused at iteration {meta_it}; Press enter to coninue.")

    if meta_it % vis_timesteps == 0:
        #Output training info to console
        print(f"Meta loop {meta_it+1}/{meta_iterations} complete, validation loss: {meta_loss.detach().item()}, validation return: {meta_ret}")

        #Qualitative plot of adapted policy
        fig, ax = plt.subplots()
        ax = env.unwrapped.show_state(ax)    
        plt.savefig(f"{model_save_path}/training{meta_it}")
        plt.clf()

        #Save meta model
        torch.save(meta_agent.policy.state_dict(), f"{model_save_path}/meta_it_{meta_it}")

torch.save(meta_agent.policy.state_dict(), f"{model_save_path}/final")

#######################################################
print("Plotting meta learning curves")
##Plot meta learning curves
xs = range(len(meta_losses))

plt.plot(xs, meta_losses)
plt.xlabel('Meta learning steps')
plt.ylabel('Validation loss')
plt.title('Meta learning curve - loss')
plt.savefig(f"{model_save_path}/trainingLossCurve")
plt.clf()

plt.plot(xs, meta_rets)
plt.xlabel('Meta learning steps')
plt.ylabel('Validation return')
plt.title('Meta learning curve - return')
plt.savefig(f"{model_save_path}/trainingReturnCurve")
plt.clf()
#######################################################
print("Loading in model")
loaded_meta_agent = A2C("MlpPolicy", env, verbose=0, learning_rate=adapt_lr, device=device,
                 meta_learning=True, M=M, adapt_timesteps=adapt_timesteps, eval_timesteps=eval_timesteps)
loaded_meta_agent.policy.load_state_dict(torch.load(f"{model_save_path}/final", weights_only=True)) 
    #can also load in intermediate saved envs from training
#######################################################
print("Qualitative evaluation of loaded model")
#Seeing it manage in a range of environments - this one is taken from loading the best model from a 400 meta-step training, which clearly isnt quite all that 
dim = 3
fig, axs = plt.subplots(dim, dim, figsize=(10, 8))

for t in tqdm(range(dim*dim)):
    #Perform few shot adaption to environment
    env.unwrapped.reset_task() #randomly selects task from environment to reset it to
    _,_,adapted_policy = loaded_meta_agent.meta_adapt(task = env.unwrapped.get_task(), M=1)
    adapted_policy = adapted_policy[0] # we set M=1 above so only 1 policy adapted here

    #Test against a new trajectory from that state (else we are showing something it trained to and before the final training step)
    loaded_meta_agent.evaluate_policy(total_timesteps=32, policy=adapted_policy)

    #Plot this run
    x = t//dim
    y = t%dim
    axs[x,y] = env.unwrapped.show_state(axs[x,y])    
    axs[x,y].set_axis_off()
    
plt.tight_layout()
plt.savefig(f"{model_save_path}/finalQualititativeEvaluation")
plt.clf()
#######################################################

#######################################################

#######################################################

#######################################################
