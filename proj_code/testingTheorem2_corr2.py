#THIS SCRIPT IS ALL ABOUT TRYING OUT MY SMC FOR THEOREM 1 AND SEEING WHAT HAPPENS!

print("THIS SCRIPT USES *WINDOWS* FILEPATHS. CHANGE FOR LINUX!! (inc the additional 'proj_code' at start for some reason)")

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


####################################################################################
print("Setting up...")
model_save_path = "C:\\Users\\fabia\\Documents\\GitHub\\MetaStableBaselines3\\proj_code\\saved_models\\19May_TestingMAML_A2C" #simulatenously testing my new maml syntax out and doing PPO (if somethings up, then try A2C with my new maml syntax to see if its my PPO that is wrong or the MAML syntax (too?))

device='cpu'
adapt_lr =  7e-4
adapt_timesteps = 32*4 #for this enviornment, each episode is exactly 32 timesteps, so multiple of 32 means full number of eps experienced for each task
eval_timesteps = 100
M=1

env = gym.make("KhazadDum-v1") # can access wrapped env with "env.unwrapped" (e.g. to reset task)
env.unwrapped.exp_bonus = 1; env.unwrapped.bridge_bonus_factor = 2 #this should incentivise getting to the target asap, and incentivise going onto the bridge


meta_agent = A2C("MlpPolicy", env, verbose=0, learning_rate=adapt_lr, device=device,
                 meta_learning=True, M=M, adapt_timesteps=adapt_timesteps, eval_timesteps=eval_timesteps)
meta_agent.policy.load_state_dict(torch.load(f"{model_save_path}\\final", weights_only=True))

####################################################################################
print("Starting SMC as a non-robust policy...")

eta = 0.9
gamma = 0.01 #small to be pretty confient in our bounds (therefore they will be pretty poor guarantees i suppose? hopefully that is what should happen -big t!)
tasks = 40
k=0

returns = meta_agent.sample_returns(tasks=tasks, repeats_per_task=10)
print(returns)


a=-1-3*(1)
b=5/env.unwrapped.normalize_rewards + 1 + env.unwrapped.bridge_bonus_factor*env.unwrapped.exp_bonus
print(f"(a,b) = {(a,b)}")

epsilon, guarantee =  meta_agent.task_risk_SMC_c2(gamma, eta, k, returns, (a,b)) #TODO: intereswtingly too conservative bounds here may make c3 better!
print((epsilon, guarantee))



bounding_returns = meta_agent.sample_returns(tasks=40, repeats_per_task=1)
bounding_returns = [i[0] for i in bounding_returns.values()]

epsilon, guarantee =  meta_agent.task_risk_SMC_c3(gamma, eta, k, bounding_returns, returns)
print((epsilon, guarantee))
