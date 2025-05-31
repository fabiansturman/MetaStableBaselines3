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


model_save_path = "saved_models/27May_MAML_A2C_KDcont3_fakeCVaR3" #simulatenously testing my new maml syntax out and doing PPO (if somethings up, then try A2C with my new maml syntax to see if its my PPO that is wrong or the MAML syntax (too?))

device='cpu'
adapt_lr =  7e-4
adapt_timesteps = 32*4 #for this enviornment, each episode is exactly 32 timesteps, so multiple of 32 means full number of eps experienced for each task
eval_timesteps = 64 # evaluate one full epsiode at a time
M=1

env = gym.make("KhazadDum-v1",continuous=True, max_speed=0.5, max_episode_steps=64) # can access wrapped env with "env.unwrapped" (e.g. to reset task)
env.unwrapped.exp_bonus = 1; env.unwrapped.bridge_bonus_factor = 2 #this should incentivise getting to the target asap, and incentivise going onto the bridge



its = [0,100,200,300]#,400,500,600,700,800,900]

for it in its:
    print(f"Getting return distribution for model at meta iteration {it}")


    meta_agent = A2C("MlpPolicy", env, verbose=0, learning_rate=adapt_lr, device=device,
                    meta_learning=True, M=M, adapt_timesteps=adapt_timesteps, eval_timesteps=eval_timesteps)
    meta_agent.policy.load_state_dict(torch.load(f"{model_save_path}/final", weights_only=True))

    print("Return generation for SMC: generating totally i.i.d returns")

    tasks=20_000#100_000

    returns = meta_agent.sample_returns(tasks=tasks, repeats_per_task=1)
    return_list = list(returns.values())

    try:
        dbfile = open(f"{model_save_path}/iidReturnList_metait{it}.pickle", 'rb')
        return_list=pickle.load(dbfile)+return_list
    except:
        print("making the file")


    dbfile = open(f"{model_save_path}/iidReturnList.pickle_metait{it}", 'wb')
    pickle.dump(return_list, dbfile)

    print("WHEN PLOTTING CAN COMPARE TO FINAL WITH 100_000 TASKS TO GET AN APPROXIMATE BEST MODEL (IK SAME MODEL BUT STILL!)")
