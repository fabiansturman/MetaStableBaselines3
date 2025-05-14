import multiprocessing as mp
import sys

import gym
import numpy as np

is_py2 = (sys.version[0] == '2')
if is_py2:
    pass
else:
    pass


class EnvWorker(mp.Process):
    def __init__(self, remote, env_fn, queue, lock):
        super(EnvWorker, self).__init__()
        self.remote = remote
        self.env = env_fn()
        self.queue = queue
        self.lock = lock
        self.task_id = None
        self.done = False

    def empty_step(self):
        observation = np.zeros(self.env.observation_space.shape,
                               dtype=np.float32)
        reward, done = 0.0, True
        return observation, reward, done, {}

    def try_reset(self):
        observation = self.env.reset()
        return observation

    def run(self):
        while True:
            command, data = self.remote.recv()
            if command == 'step':
                observation, reward, done, info = self.env.step(data)
                if done and (not self.done):
                    observation = self.try_reset()
                self.remote.send((observation, reward, done, self.task_id, info))
            elif command == 'reset':
                observation = self.try_reset()
                self.remote.send((observation, self.task_id))
            elif command == 'set_task':
                self.env.unwrapped.set_task(data)
                self.remote.send(True)
            elif command == 'close':
                self.remote.close()
                break
            elif command == 'get_spaces':
                self.remote.send((self.env.observation_space,
                                  self.env.action_space))
            else:
                raise NotImplementedError()


class SubprocVecEnv(gym.Env):
    def __init__(self, env_factory, queue):
        self.lock = mp.Lock()
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in env_factory])
        self.workers = [EnvWorker(remote, env_fn, queue, self.lock)
                        for (remote, env_fn) in zip(self.work_remotes, env_factory)]
        for worker in self.workers:
            worker.daemon = True
            worker.start()
        for remote in self.work_remotes:
            remote.close()
        self.waiting = False
        self.closed = False

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        self.observation_space = observation_space
        self.action_space = action_space

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        observations, rewards, dones, task_ids, infos = zip(*results)
        return np.stack(observations), np.stack(rewards), np.stack(dones), task_ids, infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        observations, task_ids = zip(*results)
    #    print(f"Observations length: {len(observations)}")
     #   print(f"Observations[0]: {observations[0]}")
        return np.stack(observations), task_ids

    def set_task(self, tasks):
        for remote, task in zip(self.remotes, tasks):
            remote.send(('set_task', task))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for worker in self.workers:
            worker.join()
        self.closed = True

#Fabian: Implementing synchronous versions of these evvironemnts
#TODO: get this able to be imported and then create a SyncVecEnv class too, and test it as a way to get enviornmetns working with a single worker
class SubEnv():
    """
    Objects of this class encapsulate the (meta-)envionments within a synchronous vector environment
    """
    def __init__(self, env_fn):
        self.env = env_fn()
        self.task_id = None #TODO: Not sure what this is for - it doesnt seem to be being used either here or even in the existing parelell environemtn. Was it originally there for a different sort of concurrent approach, but then in the end not needed?
        self.done = False

    def empty_step(self):
        observation = np.zeros(self.env.observation_space.shape,
                               dtype=np.float32)
        reward, done = 0.0, True
        return observation, reward, done, {}

    def try_reset(self):
        observation = self.env.reset()
        return observation
    
    #The below functions are seperated out from what was previously the 'run' function (for the asynchronous workers)
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if done and (not self.done):
            observation = self.try_reset()
        return (observation, reward, done, self.task_id, info)
    
    def reset(self):
        observation = self.try_reset()
        return (observation, self.task_id)
    
    def set_task(self, task):
        self.env.unwrapped.set_task(task)
        return True
    
    def get_spaces(self):
        return (self.env.observation_space,
                self.env.action_space)
    



class SyncVecEnv(gym.Env):
    def __init__(self, env_factory):
        self.envs = [SubEnv(env_fn) for env_fn in env_factory] 

        observation_space, action_space = self.envs[0].get_spaces()
        self.observation_space = observation_space
        self.action_space = action_space

    def step(self, actions):
        results = []
        for env, action in zip(self.envs, actions):
            results.append(env.step(action))
        observations, rewards, dones, task_ids, infos = zip(*results)
        return np.stack(observations), np.stack(rewards), np.stack(dones), task_ids, infos

    def reset(self):
        results = []
        for env in self.envs:
            results.append(env.reset())
        observations, task_ids = zip(*results)
        return np.stack(observations), task_ids

    def set_task(self, tasks):
        results = []
        for env, task in zip(self.envs, tasks):
            results.append(env.set_task(task))
        return np.stack(results)

    #No 'closing' is needed for this synchronous environemnt, but it is worth having this function for convinience, so that any meta environment classes can be 'closed' by the user at the end and then all is well and good (and if it is not needed, just nothing happens)
    def close(self):
        pass
