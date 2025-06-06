from typing import Any, ClassVar, Optional, TypeVar, Union

import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance



SelfA2C = TypeVar("SelfA2C", bound="A2C")




class A2C(OnPolicyAlgorithm):
    """
    Advantage Actor Critic (A2C)

    Paper: https://arxiv.org/abs/1602.01783
    Code: This implementation borrows code from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    and Stable Baselines (https://github.com/hill-a/stable-baselines)

    Introduction to A2C: https://hackernoon.com/intuitive-rl-intro-to-advantage-actor-critic-a2c-4ff545978752

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
    :param rms_prop_eps: RMSProp epsilon. It stabilizes square root computation in denominator
        of RMSProp update
    :param use_rms_prop: Whether to use RMSprop (default) or Adam as optimizer
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation. See :ref:`a2c_policies`
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance

    Meta-learning parameters:
    :param meta_learning: Whether to wrap the policy in a MAML module in order to calculator meta-gradients over (adaptation) training of the policy
    :param M: Number of models we adapt to the task
    :param adapt_timesteps: Number of timesteps for each adaption (proportional to the 'number of shots' in our learning, with its conventional definition as K(-shot)= # trajectories used to learn)
    :param eval_timesteps: Number of timesteps with which we evaluate how well a policy has adapted to a given task    
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 7e-4,
        n_steps: int = 5,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rms_prop_eps: float = 1e-5,
        use_rms_prop: bool = True,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        normalize_advantage: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        meta_learning = False,
        M = None,
        adapt_timesteps = None,
        eval_timesteps = None
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False, #set _init_setup_model to be False here, so that we call it only once, at the end of this constructor (as opposed to calling it also at the end of the parent constructor)
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
            meta_learning = meta_learning,
            M=M,
            adapt_timesteps=adapt_timesteps,
            eval_timesteps=eval_timesteps
        )

        self.normalize_advantage = normalize_advantage

        # Update optimizer inside the policy if we want to use RMSProp
        # (original implementation) rather than Adam
        if use_rms_prop and "optimizer_class" not in self.policy_kwargs:
            self.policy_kwargs["optimizer_class"] = th.optim.RMSprop
            self.policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99, eps=rms_prop_eps, weight_decay=0)

        if _init_setup_model:
            self._setup_model()
        
    def train(self) -> None:
        """       
            Consume current rollout data and perform single update of {self.policy} parameters.
            Uses A2C algorithm to perform this training.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        #Do a single rollout (single batch) and calculate loss over this batch
        loss, _, entropy_loss, policy_loss, value_loss = self.get_loss_rets(policy=self.policy, detailed_return=True) #TODO: note that I have taken the following two little sections out of the loop to happen after iterating over all the losses as it is only supposed to loop once anyway!

        # Optimization step
        self.policy.optimizer.zero_grad() #within the policy this is set up just on policy.parameters()
        loss.backward()

        # Clip grad norm
        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/entropy_loss", entropy_loss.item())
        self.logger.record("train/policy_loss", policy_loss.item())
        self.logger.record("train/value_loss", value_loss.item())
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

    def train_clone(self) -> None:
        """       
            Consume current rollout data and perform single update of {self.policy_clone} parameters.
            Uses A2C algorithm to perform this training, and MAML syntax (for ability to calculate gradients over the training).
        """
        assert self.policy_clone is not None

        # Switch to train mode (this affects batch norm / dropout)
        self.policy_clone.set_training_mode(True)

        #TODO: get a way to update the learning rate within metalearning adaptation too (i suppose by tweaking self.policy_clone.lr)
        
        #Do a single rollout (single batch) and calculate loss over this batch
        loss, _, entropy_loss, policy_loss, value_loss = self.get_loss_rets(policy=self.policy_clone, detailed_return=True) #TODO: note that I have taken the following two little sections out of the loop to happen after iterating over all the losses as it is only supposed to loop once anyway!

        # Optimization step
        self.policy_clone.adapt(loss, clip_norm=self.max_grad_norm) #grad norm clipping implemented myself

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/entropy_loss", entropy_loss.item())
        self.logger.record("train/policy_loss", policy_loss.item())
        self.logger.record("train/value_loss", value_loss.item())
        if hasattr(self.policy_clone, "log_std"):
            self.logger.record("train/std", th.exp(self.policy_clone.log_std).mean().item())



    def learn(
        self: SelfA2C,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 100,
        tb_log_name: str = "A2C",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        return super().learn(total_timesteps,callback,log_interval,tb_log_name,reset_num_timesteps,progress_bar)
        
    def get_loss_rets(self, policy=None, detailed_return = False):
        """
        Clear out rollout buffer, calculate total loss and return over this buffer, and return it (as a tensor for backpropagation).
        Loss determined by A2C loss function.
        
        par policy: a reference to the policy against which we are to evaluate our actions. Default: self.policy
        par detailed_return: whether to output additional optional returns (starred returns below)
        
        returns loss, mean reward over episodes, *entropy_loss*, *policy_loss*, *value_loss*
        """
        if policy is None:
            policy = self.policy


        #This will only loop once (it gets all the data at once) - as we have specified batch_size=None (see from stable_baselines3.common.buffers.RolloutBuffer)
        for rollout_data in self.rollout_buffer.get(batch_size=None):
            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions = actions.long().flatten()

            values, log_prob, entropy = policy.evaluate_actions(rollout_data.observations, actions) #a function defined for actor-crtiic policies, that estimates the value of doing a particular action
            values = values.flatten()

            # Normalize advantage (not present in the original implementation)
            advantages = rollout_data.advantages
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Policy gradient loss
            policy_loss = -(advantages * log_prob).mean()

            # Value loss using the TD(gae_lambda) target
            value_loss = F.mse_loss(rollout_data.returns, values)

            # Entropy loss favor exploration - this loss term makes us favour increasing the entropy of the action distribution, to push us away from a constant output (or a very small number of outputted actins with high prob)
                    #ent_coef = 0 by default in constructor, though TODO: maybe make this bigger?? idk
            if entropy is None:
                # Approximate entropy when no analytical form
                entropy_loss = -th.mean(-log_prob)
            else:
                entropy_loss = -th.mean(entropy)

            loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

        if not detailed_return:
            return loss, rollout_data.returns.mean()
        else:
            return loss, rollout_data.returns.mean(), entropy_loss, policy_loss, value_loss
