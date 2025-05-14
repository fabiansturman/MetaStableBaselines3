## This file defines a class to encapsulate a gymnasium environment within a gym environment

import gym
from gym.envs.registration import register
import gymnasium



def make_gymnasium_env(env_name, **kwargs):
    """
    A replacement function for gymnasium.make, that makes a gymnasium environment and immediately converts it to a gym env
    """
    return EncapsulateGymnasium(gymnasium.make(env_name, **kwargs))

#TODO: I am not sure how I can go about registering the environemtn as the class isnt a hard coded class that is then used as an entry point...
        #... so perhaps I can just make a gymansium enviornemt (that perhaps probs has been reitgstered) and then use gymnasium_to_gym to make the env 

class EncapsulateGymnasium(gym.Env):
    """
    Using this class, we can create a gymnasium env (with wrappers as desired), encapsulate it in this class, and then use it like a gym enviornment
    When making a new environment, we need to register it in gym with:
        
        from gym.envs.registration import register
        register(
            id=X,
            entry_point=,
            [further arguments as desired]
            )
    """
    def __init__(self, wrapped_env: gymnasium.Env):
        self.wrapped_env = wrapped_env

    def _get_obs(self):
        return self.wrapped_env._get_obs()
    
    def _get_info(self):
        return self.wrapped_env._get_info()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        observation, info = self.wrapped_env.reset(seed, options)
        return observation, info
    
    def step(self, action):
        observation, reward, terminated, truncated, info = self.wrapped_env.step(action)
        return observation, reward, terminated or truncated, info
    
    def render(self):
        return self.wrapped_env.render()
    
