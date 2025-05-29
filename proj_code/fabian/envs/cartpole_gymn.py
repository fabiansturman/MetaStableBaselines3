## Credit: adapted from MuJoCo Inverted-Pendulum-v5 environment

#can run this on (stb3python) conda env (and presumably (learn2learn_env2) as well!)

import random
import numpy as np
from scipy import stats
import gymnasium as gym
from gymnasium import core, spaces
from gymnasium.utils import seeding

import matplotlib.pyplot as plt

from tqdm import tqdm

#print("NEED TO UNCOMMENT THIS WHEN ACTUALLY USING IT")
from learn2learn.gym.envs.meta_env import MetaEnv


from gymnasium.wrappers import RecordVideo





#A version of gymnasium environments where each action has some level of random offset, which defines the task

"""
i need something wherer different tasks have different levels of risks and we can make decisions which are best with the less risky tasks
if we jusat add offset to everythig then i suppose tasks with the bigger offset have a bigger risk, so then we try and make small adjustmennts maybe? But then we need smth where small adjustments dont hurt - in cartpole they may still hurt. so maybe someother env. stll this is porosb worth making
"""


class MuJoCo_ActionOffsets(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                 gymnasium_env_name,
                 offset_variance=0.1, #Actions have offset sampled for each task as N(0,offset_variance) 
                    #TODO: also probs should do smth where the actions are noisy with teh level of noise determined by task too?
                 **kwargs #extra arguments for making the gym env
                 ):
        
        self.env = gym.make(gymnasium_env_name, render_mode='rgb_array',
                                  **kwargs)
        
        
        self.offset_variance = offset_variance
        


        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.spec = self.env.spec
        self.metadata = self.env.metadata
        self.np_random = self.env.np_random
        
        self.task=None #will contian a single float which is the offset to every input

        

        #print("NEED TO UNCOMMENT THIS WHEN ACTUALLY USING IT")
        MetaEnv.__init__(self) #samples a task and sets it to self.task!


    # def seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     self.rng = np.random.RandomState(seed)
    #     return [seed]

    def close(self):
        self.env.close()
 
    def reset(self, seed=None, options=None):
        #Resets the environment without resetting the task
        return self.env.reset()

    def step(self, action):

        action += self.task #TODO: hoping this works provided actions are cont?? (i..e box?)
        
        return self.env.step(action) #is it the case that this can only be truncated not terminated? so just reward accumulated if we get to the end early or?

 


    def add_wrapper(self, wrapper, **kwargs):
        """
        Wraps the encapsulated gymnasium environment in {wrapper}
        """
        self.env = wrapper(self.env, **kwargs)
        return self
    
    def normalise_actions(self):
        self.env = NormalizeActionWrapper(self.env)
        return self
    
    def add_recorder(self, video_folder, name_prefix):
        """
        Add a wrapper to record episodes
        """
        self._recordVideo=False #TODO: maybe needs ot be a a list so that it is used as a refernec and not resloved to false immedciately, tho i dont think pyuthon will do taht anyway
        self.add_wrapper(RecordVideo,
                    video_folder=video_folder,
                    name_prefix=name_prefix,
                    episode_trigger=lambda x: self._recordVideo) #TODO: i thiknk porbably i will just need to pass in the epsidoe 
        return self
    
    def set_recording(self, record = True):
        self._recordVideo = record
    
    def get_recording(self):
        return self._recordVideo



    #-----Dealing with tasks ----- 



    def set_task(self, task):
        if isinstance(task, np.ndarray) or  isinstance(task, list):
            task = task[0]
        self.task = task 

    def get_task(self):
        return self.task

    def sample_task(self):
        """
        For each dimension in an action, an offset is sampled from N(0,self.action_variance) #TODO: pretty crude, cpould normalise based on size of action or smth, then again ig the agenbt could learn but still hmm
        """
        return random.normalvariate(0,self.offset_variance) #TODO: could also self.action_space.sample() to just sample action offsets too

    def sample_tasks(self, n_tasks):
        return [self.sample_task() for _ in range(n_tasks)]

    def reset_task(self, task=None):
        #Resets the task and the envioronment
        if task is None:
            task = self.sample_task()
        self.set_task(task)
        self.reset()

#Register this with gym
#from gymnasium.envs.registration import register

#register(
#    id='KhazadDum-v1',
#    entry_point='fabian.envs.khazad_dum_gymn:KhazadDum'
#)


#The below is taken from the gymnasium wrappers tutorial

class NormalizeActionWrapper(gym.Wrapper):
    """
    :param env: (gymnasium.Env) Gymmasium environment that will be wrapped
    """

    def __init__(self, env):
        # Retrieve the action space
        action_space = env.action_space
        assert isinstance(
            action_space, gym.spaces.Box
        ), "This wrapper only works with continuous action space (spaces.Box)"
        # Retrieve the max/min values
        self.low, self.high = action_space.low, action_space.high

        # We modify the action space, so all actions will lie in [-1, 1]
        env.action_space = gym.spaces.Box(
            low=-1, high=1, shape=action_space.shape, dtype=np.float32
        )

        # Call the parent constructor, so we can access self.env later
        super(NormalizeActionWrapper, self).__init__(env)

    def rescale_action(self, scaled_action):
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)
        :param scaled_action: (np.ndarray)
        :return: (np.ndarray)
        """
        return self.low + (0.5 * (scaled_action + 1.0) * (self.high - self.low))

    def reset(self, **kwargs):
        """
        Reset the environment
        """
        return self.env.reset(**kwargs)

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float,bool, bool, dict) observation, reward, final state? truncated?, additional informations
        """
        # Rescale action from [-1, 1] to original [low, high] interval
        rescaled_action = self.rescale_action(action)
        obs, reward, terminated, truncated, info = self.env.step(rescaled_action)
        return obs, reward, terminated, truncated, info
    


if __name__ == '__main__':
    env = MuJoCo_ActionOffsets(gymnasium_env_name='Humanoid-v5',#'Ant-v5', 
                               offset_variance=0.1,
                               )

    env.add_recorder('proj_code\\videos\\testing\\29May', 't8_')
    
    env.set_recording(True)
    
    for task in [env.sample_tasks(1)[0]]:
        env.set_task(task)
        env.reset()
        for i in tqdm(range(2000)): #default episode length is 32 steps, so if we pick a number less than this then this fails when we try to show the state (as it doesnt seem to be set up to show intermediate states)
            action = env.action_space.sample()
            obs, rew, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
    
    env.set_recording(False)

    for task in [env.sample_tasks(1)[0]]:
        env.set_task(task)
        env.reset()
        for i in tqdm(range(2000)): #default episode length is 32 steps, so if we pick a number less than this then this fails when we try to show the state (as it doesnt seem to be set up to show intermediate states)
            action = env.action_space.sample()
            obs, rew, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

    env.set_recording(True)

    for task in [env.sample_tasks(1)[0]]:
        env.set_task(task)
        env.reset()
        for i in tqdm(range(2000)): #default episode length is 32 steps, so if we pick a number less than this then this fails when we try to show the state (as it doesnt seem to be set up to show intermediate states)
            action = env.action_space.sample()
            obs, rew, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
    
    env.close() #else error when procerssing vid (i think?)

    print("complete :)")
            
        


