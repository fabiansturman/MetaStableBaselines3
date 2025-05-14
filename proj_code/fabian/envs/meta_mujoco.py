##This file offers a way to perform meta learning on mujoco environments

import gym.spaces
import gymnasium
import gym
import numpy as np

from learn2learn.gym.envs.meta_env import MetaEnv

#Function to convert from gymnasium spaces to gym spaces (as else, assertions fail in used libraries that rely on gym)
    #inspired by https://shimmy.farama.org/_modules/shimmy/openai_gym_compatibility/#_convert_space, which does the inverse conversion
def convert_space(space: gymnasium.Space) -> gym.Space:
    if isinstance(space, gymnasium.spaces.Discrete):
        return gym.spaces.Discrete(n=space.n)
    elif isinstance(space, gymnasium.spaces.Box):
        return gym.spaces.Box(low=np.array(space.low), high=np.array(space.high), dtype=space.dtype) #the box stores a numpy array of all the lower and upper bounds, so no need to specify shape
    elif isinstance(space, gymnasium.spaces.MultiDiscrete):
        return gym.spaces.MultiDiscrete(nvec=space.nvec)
    elif isinstance(space, gymnasium.spaces.MultiBinary):
        return gym.spaces.MultiBinary(n=space.n)
    elif isinstance(space, gymnasium.spaces.Tuple):
        return gym.spaces.Tuple(spaces=tuple(map(convert_space, space.spaces)))
    elif isinstance(space, gymnasium.spaces.Dict):
        return gym.spaces.Dict(spaces={k: convert_space(v) for k, v in space.spaces.items()})
    elif isinstance(space, gymnasium.spaces.Sequence):
        return gym.spaces.Sequence(space=convert_space(space.feature_space))
    elif isinstance(space, gymnasium.spaces.Graph):
        return gym.spaces.Graph(
            node_space = convert_space(space.node_space),
            edge_space = convert_space(space.edge_space)
        )
    elif isinstance(space, gymnasium.spaces.Text):
        return gym.spaces.Text(
            max_length = space.max_length,
            min_length = space.min_length,
            charset = space._char_str
        )
    else:
        raise NotImplementedError(
            f"Cannot convert space of type {space}."
        )


#TODO: can i generalise this to take any gymnasium jujoco9 environemtn name and it be a gym environemt for it?
    #then i dont need my encapsulate gymansium thig
        #^ at the very least, i can make a base calss and then change a few things perhaps the task sampling and setting?

class MetaMujoco(MetaEnv):
    def __init__(self, task=None, gymnasium_env_name = 'HalfCheetah-v5'):
        self.env = gymnasium.make(gymnasium_env_name, 
                                  reset_noise_scale = 0.1, 
                                  render_mode='rgb_array')

        self.action_space = convert_space(self.env.action_space)
        self.observation_space = convert_space(self.env.observation_space)


        MetaEnv.__init__(self, task)

class MetaCheetah(MetaEnv, gym.utils.EzPickle):
    """
    This environment is for training the MuJoCo half-cheetah to run forwards/backwards.
    Reward at each time step is equal to the average velocity in the desired direction, minus the control cost.
    Tasks are distributed {-1,1}<-Bernoulli(0.5) where +1 = 'move forward', -1 = 'move backward'
    
    This meta-environment is constructed by encapsulating a full gym environment, and tasks vary the parameters within the encapsulated environment.

    **Credit**

    Adapted from the learn2learn implementation, which itself credits Jonas Rothfuss' implementation.

    **References**

    1. Finn et al. 2017. "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks." arXiv [cs.LG].
    2. Rothfuss et al. 2018. "ProMP: Proximal Meta-Policy Search." arXiv [cs.LG].

    """

    def __init__(self, task=None, gymnasium_env_name = 'HalfCheetah-v5'):
        #Encapsulate a standard gymnasium enviornment in this gym meta-environment 
        self.env = gymnasium.make(gymnasium_env_name, 
                                  reset_noise_scale = 0.1, # Controls scale of random pertubations to inital state; default for MuJoCo = 0.1
                                  render_mode='rgb_array')
                #Note that the random pertubations to inital state are not defined by a specific 'task'; they are within each task and just are the randomness in starting states. 
                    #... if we wanted, we could have this variation of starting state also vary by task, but this probably isnt too helpful??
                        #TODO: think about this!

        self.action_space = convert_space(self.env.action_space)
        self.observation_space = convert_space(self.env.observation_space)

       # print(f"observation space: {self.observation_space}")

        MetaEnv.__init__(self, task)

    #TODO: add a function called add_wrapper, which takes a wrapper and wraps the encapsulated environemnt in it

    def add_wrapper(self, wrapper, **kwargs):
        """
        Wraps the encapsulated gymnasium environment in {wrapper}
        """
        self.env = wrapper(self.env, **kwargs)
        return self

    # -------- MetaEnv Methods --------
    def set_task(self, task):
        MetaEnv.set_task(self, task) #sets self._task =task
        self.env._forward_reward_weight = task['direction'] #sets the reward function s.t. it rewards movement in the desired direction

    def sample_tasks(self, num_tasks: int): #sample a set of {num_tasks} tasks, to then one-by-one set our environment to be and train to
        directions = np.random.choice((-1.0, 1.0), (num_tasks,))
        tasks = [{'direction': direction} for direction in directions]

        return tasks

    # -------- Mujoco Methods --------
    def _get_obs(self):
        print(self.env)
        return self.env._get_obs()
    #    return np.concatenate([
     #       self.sim.data.qpos.flat[1:],
      #      self.sim.data.qvel.flat,
       #     self.get_body_com("torso").flat,
        #]).astype(np.float32).flatten()

    # -------- Gym Methods --------
    def step(self, action): 
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated or truncated, info

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs) 
        return obs

    


if __name__ == '__main__':
    from gymnasium.wrappers import RecordVideo

    env = MetaCheetah()
    env.add_wrapper(RecordVideo,
                    video_folder="videos\\testing\\cheetah",
                    name_prefix="t1_",
                    episode_trigger=lambda x: True)


    for task in [env.get_task(), env.sample_tasks(1)[0]]:
        env.set_task(task)
        env.reset()
        for i in range(100):
            action = env.action_space.sample()
            env.step(action)
