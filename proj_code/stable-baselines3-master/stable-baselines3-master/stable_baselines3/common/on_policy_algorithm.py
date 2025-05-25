import sys
import time
import warnings
from typing import Any, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv


from learn2learn.algorithms.maml import MAML

SelfOnPolicyAlgorithm = TypeVar("SelfOnPolicyAlgorithm", bound="OnPolicyAlgorithm")


class OnPolicyAlgorithm(BaseAlgorithm):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param supported_action_spaces: The action spaces supported by the algorithm.

    Meta-learning parameters:
    :param meta_learning: Whether to wrap the policy in a MAML module in order to calculator meta-gradients over (adaptation) training of the policy
    :param M: Number of models we adapt to the task
    :param adapt_timesteps: Number of timesteps for each adaption (proportional to the 'number of shots' in our learning, with its conventional definition as K(-shot)= # trajectories used to learn)
    :param eval_timesteps: Number of timesteps with which we evaluate how well a policy has adapted to a given task
    """

    rollout_buffer: RolloutBuffer
    policy: ActorCriticPolicy

    def __init__(
        self,
        policy: Union[str, type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        rollout_buffer_class: Optional[type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[tuple[type[spaces.Space], ...]] = None,

        meta_learning = False,
        M = None,
        adapt_timesteps = None,
        eval_timesteps = None
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            support_multi_env=True,
            monitor_wrapper=monitor_wrapper,
            seed=seed,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
        )

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer_class = rollout_buffer_class
        self.rollout_buffer_kwargs = rollout_buffer_kwargs or {}

        self.meta_learning = meta_learning
        if self.meta_learning:
            #If doing meta learning, we need to ensure that the parameters for our meta learning are defined
            assert M is not None
            assert adapt_timesteps is not None
            assert eval_timesteps is not None
            
            #Save meta learning parameters
            self.M = M
            self.adapt_timesteps = adapt_timesteps
            self.eval_timesteps = eval_timesteps


        if _init_setup_model:
            self._setup_model()


    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if self.rollout_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                self.rollout_buffer_class = DictRolloutBuffer
            else:
                self.rollout_buffer_class = RolloutBuffer

        self.rollout_buffer = self.rollout_buffer_class(
            self.n_steps,
            self.observation_space,  # type: ignore[arg-type]
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            **self.rollout_buffer_kwargs,
        )
        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space, self.action_space, self.lr_schedule, use_sde=self.use_sde, **self.policy_kwargs
        )
        self.policy = self.policy.to(self.device)
        # Warn when not using CPU with MlpPolicy
        self._maybe_recommend_cpu()

        #Wrap policy in a MAML wrapper for ability to calculate meta-gradients if doing meta learning with this agent
        if self.meta_learning:
            self.policy = MAML(self.policy, self.learning_rate)  #TODO: make the learning rate adhere to a schedule and all training arguyments giuven in constructuror also be able to apply to meta learning, etc, like they would be otherweise (though it is less of a big deal here as in MAML we do few shot learening anyway)


    def _maybe_recommend_cpu(self, mlp_class_name: str = "ActorCriticPolicy") -> None:
        """
        Recommend to use CPU only when using A2C/PPO with MlpPolicy.

        :param: The name of the class for the default MlpPolicy.
        """
        policy_class_name = self.policy_class.__name__
        if self.device != th.device("cpu") and policy_class_name == mlp_class_name:
            warnings.warn(
                f"You are trying to run {self.__class__.__name__} on the GPU, "
                "but it is primarily intended to run on the CPU when not using a CNN policy "
                f"(you are using {policy_class_name} which should be a MlpPolicy). "
                "See https://github.com/DLR-RM/stable-baselines3/issues/1245 "
                "for more info. "
                "You can pass `device='cpu'` or `export CUDA_VISIBLE_DEVICES=` to force using the CPU."
                "Note: The model will train, but the GPU utilization will be poor and "
                "the training might take longer than on CPU.",
                UserWarning,
            )

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
        policy = None, #Sepcifies the policy from which we collect rollouts
                            #^by default, we collect rollouts from self.policy, but if specified it can also be self.policy_clone
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"

        if policy is None:
            policy = self.policy


        # Switch to eval mode (this affects batch norm / dropout)
        policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstrapping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Consume current rollout data and perform single update of {self.policy} parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError
    
    def train_clone(self) -> None:
        """
        Consume current rollout data and perform single update of {self.policy_clone} parameters.
        Performs training with the MAML update syntax, and adapts the clone.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def dump_logs(self, iteration: int = 0) -> None:
        """
        Write log.

        :param iteration: Current logging iteration
        """
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        if iteration > 0:
            self.logger.record("time/iterations", iteration, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        self.logger.dump(step=self.num_timesteps)

    def get_loss_rets(self, policy=None, detailed_return = False): #TOOO: was just called 'get_loss' so will need top update references to this function in my scripts
        """
        Clear out rollout buffer, calculate total loss and return over this buffer, and return it (as a tensor for backpropagation).
        Implemented specific to the algorithm being used.
        
        par policy: a reference to the policy against which we are to evaluate our actions. Default: self.policy (this defaulting must be done by implementations)
        par detailed_return: whether to output additional optional returns (depend on specific implementation)
        
        returns loss, mean reward over episodes, *[OPTIONAL RETURNS] if {detailed return}*
        """
        return NotImplementedError

    def evaluate_policy( #TODO: was run_meta_adaption_and_loss; need to change its usage in scripts!
            self: SelfOnPolicyAlgorithm,
            total_timesteps: int,
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
            policy = None):
        """
        Runs {policy} on the current enviornment {self.env} for {total_timesteps} timesteps, and returns the total loss

        If {policy} is left as None, it defaults to {self.policy_clone}. This is s.t. the returned loss can be then backpropagated w.r.t self.policy's parameters for meta learning
        
        returns (mean loss,mean return) 
        """
        #If policy not specified, try to test policy clone else the main policy itself
        if policy is None and self.policy_clone is not None:
            policy = self.policy_clone 
        elif policy is None and  self.policy_clone is None :
            policy = self.policy

        
        assert self.env is not None
        assert policy is not None 
        
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            reset_num_timesteps,
            progress_bar
        )

        iteration = 0
        loss = 0
        rets = 0

        while self.num_timesteps < total_timesteps:
            cont = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps, policy=policy)

            if not cont:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)


            delta_loss, delta_rets =  self.get_loss_rets(policy)
            loss += delta_loss
            rets += delta_rets

        return loss/iteration, rets/iteration

        

    def learn(
        self: SelfOnPolicyAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfOnPolicyAlgorithm:
        #If we are meta learning, a clone of {self.policy} will learn responses to {self.env}, else {self.policy} will learn it
        if self.meta_learning:
            self.policy_clone = self.policy.clone()
            policy = self.policy_clone
        else:
            policy = self.policy



        #Now, perform learning with {policy} (either {self.policy} itself, or our fresh policy clone as applicable to whether we are meta-learning or not)
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps, policy=policy)

            if not continue_training:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                self.dump_logs(iteration)

            
            #train:empty the filled rollout buffer and perform update step(s) based on this, according to the algorithm implementation 
            if not self.meta_learning:
                self.train() #if standard learning, then just train {self.policy}
            else:
                self.train_clone() #if meta learning, train {self.policy_clone} to adapt this clone to the enviornment

        callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []

    def meta_adapt(self, task=None, M=None, adapt_timesteps = None, eval_timesteps = None):
        """
        Adapts M fresh clones of {self.policy} to task {task}, and then evaluating the adaptations and returning mean loss and return for adaptation, alongside the adapted models themselves too.
        As M->infinity, this estimates the quality of the meta policy for adapting to task {task}.

        :param task: Task we are to adapt to; if unspecified, we randomly generate a new one from the enviornment
        :param M: Number of models we adapt to the task
        :param adapt_timesteps: Number of timesteps for each adaption (proportional to the 'number of shots' in our learning, with its conventional definition as K(-shot)= # trajectories used to learn)
        :param eval_timesteps: Number of timesteps with which we evaluate how well a policy has adapted to a given task

        :return: empirical mean loss, empirical mean return, [adapted models]

        """
        #This function is for meta learning only
        assert self.meta_learning

        if task is None: 
            #If task is not specified, randomly reset task from environment
            self.env.env_method("reset_task")
            task = self.env.env_method("get_task")[0] #when calling a function of a wrapped vectorised environment, we get a list of values back - but we just have one env here (TODO: confirm this will always be the case) so just get 0th
        else:
            #If task is specified, set the task of the enviornment to be that specified 
            self.env.env_method("reset_task", task=task)
        
        #Set any unspecified meta learning parameters to be the defaults for this agent
        if M is None: M = self.M
        if adapt_timesteps is None: adapt_timesteps = self.adapt_timesteps
        if eval_timesteps is None: eval_timesteps = self.eval_timesteps

        #Perform adaptation M times
        mean_loss = 0; mean_ret = 0; adapted_models = []
        for m in range(M):
            self.env.env_method("reset_task", task=task)#TODO: if this func is not working, could it be that the env we are dealing with is not self.env but a copy somehow?
            self.learn(total_timesteps=adapt_timesteps) #make a clone of {self.policy} and adapt this clone to the current task
            adapt_loss, adapt_ret = self.evaluate_policy(total_timesteps=eval_timesteps, policy=self.policy_clone) #sample {eval_timesteps} timesteps from the task using the adapted policy, and return over resultant path and loss from it
            
            adapted_models.append(self.policy_clone)
            mean_loss += adapt_loss; mean_ret += adapt_ret #NOTE (space efficiency): as these losses get added, the adapted models are kept in memory until we detach the loss/ret values from the computation graph
        
        mean_loss/=M; mean_ret/=M

        return mean_loss, mean_ret, adapted_models
    
    def sample_returns(self, tasks=1, repeats_per_task:int = 1, adapt_timesteps = None, eval_timesteps=None):
        """
        :param tasks: Either the number of tasks to sample (within which to sample rollouts), or a list of tasks to use
        :param repeats_per_task: Number of repeated returns to sample from each task
        :param adapt_timesteps: Number of environment timesteps with which to adapt our meta-policy to a policy instance if meta-learning (if None and metalearning, defaults to self.adapt_timesteps) 
        :param eval_timesteps: Number of environmnet timesteps with which to evaluate our (potentially adapted) policy. If not metalearning, must be specified (as it defines the number of timesteps over which we add our return), if metalearning and None, then defaults to self.eval_timesteps

TODO: I am currently not doing return based on return per rollout, instead I am doing return as defined by the functions like self.evaluate_policy and self.meta_adapt (which uses evaluate_policy). So that means my SMC is based on return over however many timesteps are specified in eval_timesteps. If tasks have fixed length then we can make it so that each loss is an single eopisodes return,but otherwise we essentailly are redefining a return to be o er a certain umber of timsteps so it all works out!!

        :return: dictionary {task: [returns from rollouts from this task]}
        """
        if adapt_timesteps is None:
            adapt_timesteps = self.adapt_timesteps
        if eval_timesteps is None:
            eval_timesteps = self.eval_timesteps
            assert eval_timesteps is not None

        #Generate tasks if only number of desired tasks given
        if isinstance(tasks, int):
            tasks = self.env.env_method("sample_tasks", n_tasks = tasks)
        assert isinstance(tasks, list)

        #Collect returns
        returns = {task:[] for task in tasks}
        for task in tasks:
            task_returns = []
            for _ in range(repeats_per_task):
                if self.meta_learning:
                    #If metalearning, we adapt to our task and get the return from that!
                    _, ret, _ = self.meta_adapt(task=task, M=1, adapt_timesteps=adapt_timesteps, eval_timesteps=eval_timesteps)
                    task_returns.append(ret)
                else:
                    #If not metalearning, immediately evaluate {self.policy} once for our task
                    self.env.env_method("reset_task", task=task)
                    _, ret = self.evaluate_policy(total_timesteps=eval_timesteps, policy=self.policy)
                    task_returns.append(ret)
            returns[task] = task_returns

        return returns
        



    def rollout_risk_SMC(self, confidence_level, no_tasks, k=0, policy = None, adapt_timesteps = None, eval_timesteps = None):
        """
        For {policy}, find risk bound \epsilon and performance guarantee \tilde{J} s.t. {policy} is certifiably robust with {confidence_level}.
        Risk function is rollout risk r_R (see Chapter 2), and bounds are computed with direct Scenario Approach method (see Chapter 3)

        :param confidence_level: Confidence level we desire for our PAC bound
        :param no_tasks: Number of tasks we will sample from (1 return-sample for each task) to find PAC bound
        :param k: we base our SMC bound on the kth order statistic of these returns. 0th order statistic=min
        :param adapt_timesteps: Number of environment timesteps with which to adapt our meta-policy to a policy instance if meta-learning (if None and metalearning, defaults to self.adapt_timesteps) 
        :param eval_timesteps: Number of environmnet timesteps with which to evaluate our (potentially adapted) policy. If not metalearning, must be specified (as it defines the number of timesteps over which we add our return), if metalearning and None, then defaults to self.eval_timesteps

        :return: risk bound \epsilon, performance guarantee \tilde{J} 
        """
        #Collect {no_tasks} returns
        returns = self.sample_returns(tasks=no_tasks, repeats_per_task=1, adapt_timesteps=adapt_timesteps, eval_timesteps=eval_timesteps)
        returns = [rets[0] for task, rets in returns] #turn {returns} to be just a list of returns
        returns = sorted(returns)
        kth_order_stat = returns[k]

        raise NotImplementedError
    

    