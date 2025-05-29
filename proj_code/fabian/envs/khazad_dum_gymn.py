## Credit: adapted from RmRL paper's code (https://github.com/ido90/RobustMetaRL/blob/main/environments/navigation/KhazadDum.py) 

#TThis version is buuilt for gymnasium

#can run this on (stb3python) conda env (and presumably (learn2learn_env2) as well!)

import random
import numpy as np
from scipy import stats
import gymnasium as gym
from gymnasium import core, spaces
from gymnasium.utils import seeding

import matplotlib.pyplot as plt

from learn2learn.gym.envs.meta_env import MetaEnv






#TODO: make a gymnasium version of meta envrionemtns (maybe a diff architecture for how they work tho)

class KhazadDum(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, task = None, 
                 action_noise=0.1, noise_offset = 0, 
                 max_episode_steps=32, continuous=False, max_speed=1,
                 obs_level=1, one_hot=True, seed=None, init_state=None, safe_banks=True,
                 exp_bonus=0, per_action_bonus=True, bridge_bonus_factor=1, eval_mode=None,
                 normalize_rewards=None, continuous_rewards=True, size=(11,9),
                 plot_interval = None #if set to None, doesnt record any episodes. Else, plots every {plot_interval} episode
                 ):

        self.continuous = continuous
        self.obs_level = obs_level  # 0 = x,y; 1 = x,y,abyss_distances
        self.one_hot = one_hot
        if isinstance(size, int):
            size = (size, size)
        self.W, self.H = size
        self.init_state_range = ((0.5,2.5), (0.5,2.5))
        self.goal_state = np.array((2, self.H-2))
        self.abyss_range = (3, 6)
        self.bridge1 = (2, 3)
        self.bridge2 = (self.W-4, self.W-1)
        self.obs_range = 2
        self.max_speed = max_speed # 1 if self.continuous else 2

        self.task_dim = 1
        self.average_noise = action_noise #this defines the distribution of the task
        self.noise_offset = noise_offset
        self.task = action_noise


        self.safe_banks = safe_banks
        self._max_episode_steps = max_episode_steps
        self.continuous_rewards = continuous_rewards
        self.normalize_rewards = max_episode_steps if normalize_rewards is None else normalize_rewards
        self.exp_bonus = exp_bonus #the multiplier for our exp_bonus - a bonus added to the reward of each step for which we are not in the abyss, which decreases at an exponential rate over time
        self.per_action_bonus = per_action_bonus
        self.bridge_bonus_factor = bridge_bonus_factor #if doing exp_bonus, this is a multiplier to the each reward bonus for when we are on the thin bridge. As bonus values decrease over time, this encourages us to be on the thin bridge
        self.suspend_bonuses = False

        # x, y, abyss-distance to l/r/u/d
        shape = self.W*self.H if self.one_hot else \
            {0:2, 1:6, 2:10, 3:16, 4:27}[self.obs_level]
        self.observation_space = spaces.Box(
            low=np.float32(-1), high=np.float32(1),
            shape=(shape,), dtype=np.float32)

        n_speeds = 1
        if self.continuous:
            # dx + dy
            self.action_space = spaces.Box(np.float32(-1), np.float32(1),
                                           shape=(2,), dtype=np.float32)
        else:
            # direction (4) X acceleration (3)
            # self.action_space = spaces.Discrete(4*3)
            self.action_space = spaces.Discrete(4*n_speeds)

        self.directions_map = [np.array((-1, 0)), np.array((1, 0)),
                               np.array((0, -1)), np.array((0, 1))]  # l, r, d, u
        # self.speed_map = [self.max_speed/4, self.max_speed/2, self.max_speed]
        # self.actions_map = lambda i: (self.directions_map[i//3], self.speed_map[i%3])
        self.speed_map = [self.max_speed/(2**i) for i in range(n_speeds)]

        #def a_m(i):
         #   
        #a_m=1
        self.n_speeds = n_speeds
        #self.actions_map = a_m #lambda i: 

        self.map = self._set_walls()
        self.goal_cell, self.goal = self._set_in_map(self.goal_state)
        self.goal_cell = np.round(self.goal_cell).astype(int)

        self.state_xy, self.state = self._set_in_map(init_state)
        self.state_cell = np.round(self.state_xy).astype(int)
        if init_state is not None:
            self.reset_state_cell = self.state_cell.copy()
            self.reset_state = self.state.copy()
        else:
            self.reset_state_cell, self.reset_state = None, None
        self.state_traj = None
        self.action_traj = None
        self.visit_count = None
        self.reset_visit_count()

        self.nsteps = 0
        self.speed = 0
        self.path = 'stay'
        self.tot_reward = 0
        self.reached_dest = 0

        self._time = 0
        self._return = 0
        self._last_return = 0
        self._curr_rets = []

        self.info = {}


        self.plot = True if plot_interval is not None else False
        if self.plot:
            self.plot_interval = plot_interval
            self.plot_interval_prog = 0

        MetaEnv.__init__(self, task)

    def actions_map(self, i):
        return (self.directions_map[i//self.n_speeds], self.speed_map[i%self.n_speeds])


    # def seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     self.rng = np.random.RandomState(seed)
    #     return [seed]

    def _get_obs(self):
        return self.get_obs()
    
    def _get_info(self):
        return self.info

    def get_obs(self):
        if self.one_hot:
            return self.one_hot_encoding()
        obs = np.array(self.state_xy / np.array((self.W,self.H)), dtype=np.float32)
        if 1 <= self.obs_level <= 3:
            obs = np.concatenate((
                obs, np.array(self.abyss_distances(
                    self.obs_level >= 2, self.obs_level >= 3), dtype=np.float32)))
        elif self.obs_level == 4:
            obs = np.concatenate((self.state_xy-self.state_cell,
                                  self.get_local_map(2).reshape(-1)), dtype=np.float32)
        return obs

    def get_local_map(self, rad=2):
        pad = max(0, rad - 1)
        nw = self.W + 2*pad
        nh = self.H + 2*pad
        padded_map = np.ones((nw, nh), dtype=np.float32)
        padded_map[pad:nw-pad, pad:nh-pad] = self.map  # 1=wall, -1=abyss
        x = pad + self.state_cell[0]
        y = pad + self.state_cell[1]
        return padded_map[x-rad:x+rad+1, y-rad:y+rad+1]

    def one_hot_encoding(self):
        m = np.zeros((self.W,self.H), dtype=np.float32)  # i,j=0,...,n-1
        x, y = self.state_xy  # 0.5<=x,y<=n-1.5
        i0, j0 = self.state_xy.astype(int)  # 0<=i0,j0<=n-2
        i1, j1 = i0+1, j0+1
        dx, dy = i1-x, j1-y
        m[i0,j0] = dx*dy
        m[i0,j1] = dx*(1-dy)
        m[i1,j0] = (1-dx)*dy
        m[i1,j1] = (1-dx)*(1-dy)
        return m.reshape(-1)


    def abyss_distances(self, walls_distances=False, high_level_encoding=False):
        R = self.obs_range
        x, y = list(self.state_xy)

        def ldist(x0):
            return min((x-x0+0.5)/R, 1)
        def rdist(x0):
            return min((x0-x-0.5)/R, 1)
        def ddist(y0):
            return min((y-y0+0.5)/R, 1)
        def udist(y0):
            return  min((y0-y-0.5)/R, 1)


        #ldist = lambda x0: 
        #rdist = lambda x0: 
        #ddist = lambda y0: 
        #udist = lambda y0:

        lr = [1, 1]
        if self.abyss_range[0]-0.5 < y < self.abyss_range[1]-0.5:
            if x < self.bridge1[0]-0.5:
                lr = [0, 0]
            elif x <= self.bridge1[1]-0.5:
                lr = [ldist(self.bridge1[0]), rdist(self.bridge1[1])]
            elif x < self.bridge2[0]-0.5:
                lr = [0, 0]
            elif x <= self.bridge2[1]-0.5:
                lr = [ldist(self.bridge2[0]), rdist(self.bridge2[1])]
            else:
                lr = [0, 0]

        du = [1, 1]
        if not (self.bridge1[0]-0.5 <= x <= self.bridge1[1]-0.5 or
                self.bridge2[0]-0.5 <= x <= self.bridge2[1]-0.5):
            if self.abyss_range[0]-0.5 < y < self.abyss_range[1]-0.5:
                du = [0, 0]
            elif y <= self.abyss_range[0]-0.5:
                du = [1, udist(self.abyss_range[0])]
            else:
                du = [ddist(self.abyss_range[1]), 1]

        w = []
        if walls_distances:
            w.append(ldist(1))
            w.append(rdist(self.W-1))
            w.append(ddist(1))
            w.append(udist(self.H-1))

        location_category = []
        if high_level_encoding:
            location_category = 6*[0]
            location_category[self.location_category()] = 1

        return lr + du + w + location_category

    def location_category(self):
        # specify high-level location:
        #  bottom left(0)/center(1)/right(2), bridge1(3), bridge2(4), top(5)
        x, y = self.state_xy
        if y <= self.abyss_range[0]-0.5:
            if x <= self.bridge1[0]-0.5:
                return 0
            elif x <= self.bridge1[1]-0.5:
                return 1
            else:
                return 2
        elif y >= self.abyss_range[1]-0.5:
            return 5
        else:
            if x < self.bridge1[1]:
                return 3
            else:
                return 4

    def reset(self, seed=None, options=None):
        # if self.reset_state is not None:
        #     self.state_cell, self.state = self.reset_state_cell, self.reset_state
        #     self.state_xy = self.state_cell.copy()
        # else:
        #     if init_state is None:
        init_state = np.array([random.uniform(*self.init_state_range[0]),
                               random.uniform(*self.init_state_range[1])])
        self.state_xy, self.state = self._set_in_map(init_state)
        self.state_cell = np.round(self.state_xy).astype(int)

        self.curr_state_traj = [np.reshape(self.state_xy,(1,self.state_xy.size))]
        self.curr_action_traj = []
        self.nsteps = 0
        self.speed = 0
        self.tot_reward = 0
        self.path = 'stay'
        self.reached_dest = 0

        obs = self._get_obs()
        info=self._get_info()
        return obs, info

    def reset_visit_count(self):
        action_dim = 1
        if self.per_action_bonus:
            action_dim = 4 if self.continuous else self.action_space.n
        self.visit_count = np.zeros((self.map.shape[0], self.map.shape[1],
                                     action_dim), dtype=np.int32)

    def get_continuous_action_direction(self, action):
        # l, r, d, u
        dx, dy = action
        if -dx > np.abs(dy): return 0  # l
        if  dx > np.abs(dy): return 1  # r
        if -dy >= np.abs(dx): return 2  # d
        # if  dy >= np.abs(dy): return 3  # u
        return 3

    def get_exp_bonus(self, action):
        bonus = 0
        if self.exp_bonus and not self.suspend_bonuses:
            i, j = self.state_cell[0], self.state_cell[1]
            k = 0
            if self.per_action_bonus:
                if self.continuous:
                    k = self.get_continuous_action_direction(action)
                else:
                    k = action
            bonus_turns = 256 // self.visit_count.shape[2]

            bonus_factor = 1
            if self.bridge_bonus_factor!=1 and self.location_category()==3:
                bonus_factor *= self.bridge_bonus_factor

            if self.visit_count[i, j, k] < bonus_turns:
                bonus = bonus_factor * self.exp_bonus / \
                        np.sqrt(self.visit_count[i, j, k] + 1)
            self.visit_count[i, j, k] += 1

            #print(f"exp_bonus getting us bonus: {bonus}") 0<=bonus <=self.bridge_bonus_factor *self.exp_bonus / sqrt(visits to a given sate with a given action+1) <= self.bridge_bonus_factor*self.exp_bonus

        return bonus

    def get_reward(self, action, is_noisy):
        bonus = 0
        if self.reached_dest > 0:
            #print("-REACHED-")
            r = 0
        elif self.reached_dest < 0:
            #print("-ABYSS-")
            r = -1 # for every step we are in the abyss, we have a reward of -1
        else:
            #print("-NEITHER-")
            if self.continuous_rewards:
                r = max(-np.sum(np.abs(self.state_xy-self.goal_state)) / 5, -1) #if continuous rewards, we have a reward -1<=r<=0, bigger for further away from goal
            else:
                r = -1 #if discrete rewards, and not at goal, our reward is -1 to start with
           
            if is_noisy:
                r -= 3 * self.task #more noisy tasks means less reward. reward shifted down by 3*self.task

            # success
            if self.goal[self.state_cell[0], self.state_cell[1]] > 0:
                self.reached_dest = 1
                self.path += '_done'
                r += 5 #the first time we reach the goal, we get 5 added to the reward

            # abyss
            if self.map[self.state_cell[0], self.state_cell[1]] < 0:
                self.reached_dest = -1
                self.path += '_fall'
                # r -= 10

        if self.reached_dest >= 0:
            #We are eligible for the exp_bonus iff we have not fallen into the abyss. 
            

           # print("not in the abyss, so getting bonus")
            # if self.continuous_rewards:
            #     abyss_distance = np.min(self.abyss_distances())  # in [0,1]
            #     r -= (1-abyss_distance)
            bonus = self.get_exp_bonus(action)
            
                 #0<=bonus <=self.bridge_bonus_factor *self.exp_bonus / sqrt(visits to a given sate with a given action+1) <= self.bridge_bonus_factor*self.exp_bonus

        
        r += bonus
        r /= self.normalize_rewards #unless otherwise specified, this is self.max_epsiode_steps

        return r

    def step_continuous(self, action):
        # theta = 2*np.pi * action[0]
        # self.speed = self.max_speed * action[1]
        # delta = self.speed * np.array((np.sin(theta), np.cos(theta)))

        delta = self.max_speed * np.array(action)
        self.speed = np.linalg.norm(delta)

        return self.state_xy + delta

    def step_discrete(self, action):
        if isinstance(action, np.ndarray):
            action = action[0]
        direction, self.speed = self.actions_map(action)
        return self.state_xy + self.speed * direction

    def is_noisy(self):
        return self.reached_dest == 0 and self.task > 0 and (
                (not self.safe_banks) or self.location_category() == 3)

    def step(self, action):
        self.curr_action_traj.append(action)

        is_noisy = self.is_noisy()
        if self.reached_dest == 0:
            next_xy = self.step_continuous(action) if self.continuous \
                else self.step_discrete(action)
            if is_noisy:
                next_xy[0] += random.normalvariate(
                    0, self.task * self.speed**2)
                next_xy[1] += random.normalvariate(
                    0, self.task * self.speed**2)
            next_xy = np.clip(next_xy, (0, 0), (self.W-1, self.H-1))
            next_cell = np.round(next_xy).astype(int)

            # moved = False
            if self.map[next_cell[0], next_cell[1]] != 1:
                # moved = True
                self.state_xy = next_xy
                self.state_cell = next_cell
                self.state = np.zeros_like(self.map)
                self.state[self.state_cell[0], self.state_cell[1]] = 1

        self.curr_state_traj.append(np.reshape(self.state_xy,(1,self.state_xy.size)))

        r = self.get_reward(action, is_noisy)

        
        truncated = False
        if self.nsteps >= self._max_episode_steps-1:
            truncated = True


        obs = self.get_obs()

        if self.abyss_range[0] < self.state_xy[1]+0.5 < self.abyss_range[1]:
            if self.bridge1[0] < self.state_xy[0]+0.5 < self.bridge1[1]:
                self.path = 'short'
            elif self.bridge2[0] < self.state_xy[0]+0.5 < self.bridge2[1]:
                self.path = 'long'

        self.tot_reward += r
        self.nsteps += 1
        self.info = {'path': self.path}

        if self.plot:
            self.plot_interval_prog +=1

        if truncated:
            self.state_traj = self.curr_state_traj
            self.action_traj = self.curr_action_traj
            self.info = {'r': self.tot_reward, 'l': self.nsteps,
                    'path': self.path}
            
            #Plot if desired
            if self.plot:
                if self.plot_interval_prog % self.plot_interval == 0:
                    fig, ax = plt.subplots()
                    ax = self.show_state(ax)     #env.unwrapped gets the enviornemtn at the bottom of wrapper
                    fig.show()

                    self.plot_interval_prog = 0

        self._time += 1
        self._return += r
        if self._time % self._max_episode_steps == 0:
            self._last_return = self._return
            self._curr_rets.append(self._return)
            self._return = 0

        #print(f"Current location category: {self.location_category()} ")

        return obs, r, False, truncated, self.info #is it the case that this can only be truncated not terminated? so just reward accumulated if we get to the end early or?

    def _set_in_map(self, xy=None, radius=0.):
        if xy is None:
            cell = random.choice(np.arange(self.W)), random.choice(np.arange(self.H))
            while self.map[cell[0], cell[1]] != 0:
                cell = random.choice(np.arange(self.W)), random.choice(np.arange(self.H))
            xy = np.array(cell).copy()
        else:
            xy = np.array(xy)
            xy = xy % self.H
            cell = np.round(xy).astype(int)

        map = np.zeros_like(self.map)
        map[int(cell[0]-radius) : int(cell[0]+radius+1),
            int(cell[1]-radius) : int(cell[1]+radius+1)] = 1

        return xy, map

    def _im_from_state(self, for_plot=True):
        # map
        m = self.map
        if for_plot:
            m = np.array([[1.,0.,0.][int(x)] for x in m.reshape(-1)]).reshape(m.shape)
            m[self.goal_cell[0], self.goal_cell[1]] = 0.
        im_list = [0.7*m]

        # goal
        im_list.append(self.goal + 0.3*m)

        # state (agent)
        im_list.append(self.state + 0.2*m)

        im = np.stack(im_list, axis=0)

        if for_plot:
            walls = np.array([[0,0.7,0][int(x)]
                              for x in self.map.reshape(-1)]).reshape(self.map.shape)
            walls = np.stack(3*[walls], axis=0)
            im += walls

        return im

    def show_state(self, ax, color_scale=0.7, show_traj=True, traj_col='w', show_task=False, text_coords=(0,0)):
        im = self._im_from_state(for_plot=True).swapaxes(0, 2)
        im[self.state_cell[1],self.state_cell[0],2] = 0  # remove agent square
        ax.imshow(color_scale*im)
        ax.invert_yaxis()
        if show_traj:
            track = np.concatenate(self.state_traj)
            ax.plot(track[:, 0], track[:, 1], f'{traj_col}.-')
            ax.plot(track[:1, 0], track[:1, 1], f'{traj_col}>', markersize=12)
            ax.plot(track[-1:, 0], track[-1:, 1], f'{traj_col}s', markersize=10)
        if show_task:
            ax.text(text_coords[0], text_coords[1], f"{self.task}")
        return ax

    def _set_walls(self):
        W = self.W
        H = self.H
        map = np.zeros((W, H))

        # abyss
        map[:self.bridge1[0], self.abyss_range[0]:self.abyss_range[1]] = -1
        map[self.bridge1[1]:self.bridge2[0],
            self.abyss_range[0]:self.abyss_range[1]] = -1
        map[self.bridge2[1]:, self.abyss_range[0]:self.abyss_range[1]] = -1

        # outer walls
        map[0, :] = 1
        map[:, 0] = 1
        map[-1:, :] = 1
        map[:, -1:] = 1

        return map

    def render(self, mode='human', close=False):
        return 0

    def get_last_return(self):
        return np.sum(self._curr_rets)

    #-----Dealing with tasks ----- 



    def set_task(self, task):
        if isinstance(task, np.ndarray) or  isinstance(task, list):
            task = task[0]
        self.task = task 

    def get_task(self):
        return self.task

    def sample_task(self, avg=None):
        """
        Noise is exponentially distributed with mean {avg}, and then offset by {self.noise_offset} (defaults at 0 in construction)
        """
        if avg is None:
            avg = self.average_noise
        if avg == 0:
            return 0 + self.noise_offset
        return random.expovariate(1/avg) + self.noise_offset

    def sample_tasks(self, n_tasks):
        return [self.sample_task() for _ in range(n_tasks)]

    def reset_task(self, task=None):
        if task is None:
            task = self.sample_task()
        self.set_task(task)
        self._time = 0
        self._last_return = self._return
        self._curr_rets = []
        self._return = 0
        # self.reset()

#Register this with gym
from gymnasium.envs.registration import register

register(
    id='KhazadDum-v1',
    entry_point='fabian.envs.khazad_dum_gymn:KhazadDum'
)



if __name__ == '__main__':
    env = KhazadDum(continuous=True, max_speed=0.5)

    
    for task in [env.get_task(), env.sample_tasks(1)[0]]:
        env.set_task(task)
        env.reset()
        for i in range(32): #default episode length is 32 steps, so if we pick a number less than this then this fails when we try to show the state (as it doesnt seem to be set up to show intermediate states)
            action = env.action_space.sample()
            env.step(action)
            
        


    fig, ax = plt.subplots()
    ax = env.show_state(ax)    
    fig.show()
    input()
    #plt.savefig('plot.png')