import logging as log
from datetime import datetime as dt
from multiprocessing import Pool

import gym
import numpy as np
import pandas as pd
from numpy.random import RandomState


class RandomPolicy(object):

    def __init__(self, possible_actions=range(6), random_seed=0):
        """
        A policy that will take random actions actions.
    
        Arguments: 
        ----------
        possible_actions: list
            A list of possible actions to sample from.
        random_seed: int
            The seed for the random number generator
        """
        self.random_state = RandomState(random_seed)
        self.possible_actions = possible_actions
        
    def get_action(self, observation, reward):
        """
        sample a random action.
    
        Arguments of this function are not used but exist to allow
        easy switching to a policy that makes use of the obs.
        
        Arguments:
        ----------
        observation: Numpy Array
            As returned by gym's env.step(...)
        reward: float
            As returned by gym's env.step(...)
            
        Returns:
        --------
        action:
            One one possible_actions, uniformly sampled.
        """
        randint = self.random_state.randint
        action_id = randint(0, len(self.possible_actions))
        action = self.possible_actions[action_id]
        return action


class KerasModelPolicy(object):

    def __init__(self,
                 possible_actions=range(6),
                 random_seed=0,
                 model=None,
                 probabilistic_mode=False):
        """
        A policy that will use a Keras Model.

        The observations will be given to the Keras Model.
        The output of the model should match the shape of
        possible actions.

        Arguments:
        ----------
        possible_actions: list
            A list of possible actions to sample from.
        random_seed: int
            The seed for the random number generator
        model: Keras model
            The acutal model to translate observations
            into actions. Is expected to take a 4d
            array as (num, y, x, channel) to same shape
            as possible_actions.
        probabilistic_mode: bool
            If true will interpret the output of the
            model as probabilities of chosing the actions.
            The next action will be sampled accordingly.
            If false will choose the action with the highest
            value.
        """
        self.random_state = RandomState(random_seed)
        self.possible_actions = possible_actions
        self.model = model
        self.probabilistic_mode = probabilistic_mode

        self.black_bound_shape = model.input_shape[1:3]

    def get_action(self, observation, reward):
        """
        Compute the next action by using the keras model.

        Arguments:
        ----------
        observation: Numpy Array
            As returned by gym's env.step(...)
        reward: float
            As returned by gym's env.step(...)

        Returns:
        --------
        action: ?
            Selected action of self.possible_actions
        """
        # Process observations to match model input
        create_black_boundary = ExtractWorker.create_black_boundary
        observation = create_black_boundary([observation],
                                            self.black_bound_shape)[0]
        observation = np.expand_dims(observation, axis=0)

        # Query the model for an action
        model_output = self.model.predict(observation)[0]

        # Choose the action that has the highest output value
        if not self.probabilistic_mode:
            action_id = model_output.argmax()
            action = self.possible_actions[action_id]

        # Select next action by sampling according to the
        # probabilities predicted by the model.
        else:
            # Normalize to one, just in case.
            probabs = model_output / model_output.sum()
            choice = self.random_state.choice
            action = choice(self.possible_actions, p=probabs)

        return action


class ExtractWorker(object):
    
    def __init__(self, env_name='SpaceInvaders-v4', custom_frame_skip_length=1, 
                 observation_callback=None):
        """
        The worker process that derives episodes from gym environments
        
        Arguments:
        ----------
        env_name: String
            The name of the environment to use. As expected by gym.make
        use_custom_frame_skip: bool
            Applies self.custom_frame_skip function to custom_frame_skip_length
            outputs of env.step while repeating the action. No effect for 
            custom_frame_skip_length = 1.
        observation_callback: None or fuction
            If not None: Will be called as observation_callback(observation)
            directly after env.step returns the observations. Will be applied
            before custom_frame_skip functions gets into action.
        """
        self.env = gym.make(env_name)
        self.custom_frame_skip_length = custom_frame_skip_length
        self.observation_callback = observation_callback

    @staticmethod
    def custom_frame_skip(observations, dones, rewards, infos):
        """
        Improves visibility of laser beams.
        """
        observation = np.max(np.stack(observations, axis=0), axis=0)
        done = max(dones)
        reward = sum(rewards)
        info = infos[-1]
        
        return observation, reward, done, info

    def extract_episode(self, policy, max_steps=-1):
        """
        Extract one episode of the environment and return it.
        
        Arguments:
        ----------
        policy: object
            Will be called as policy.get_action(observation, reward) to derivce
            the action for the next steps.
            
        max_step: int or None
            If int the maximum number of steps one episode should contain, 
            including the initial state before the game starts. 
            If -1 the episode will be run until it terminates i.e. 
            env.step returns done=True
            
        Returns:
        --------
        observations: list
            list of objects returned at each call of env.step
        actions: list
            list of objects returned at each call of env.step
        rewards: list
            list of objects returned at each call of env.step
        infos: list
            list of objects returned at each call of env.step
        """
        observations = []
        actions = []
        rewards = []
        infos = []
        
        current_step = 0
        
        # This is the observation before the episode starts.
        observation = self.env.reset()
        
        # Initialise to None to make clear that this is no output 
        # of gym.env or anything we computed (w.r.t. the action)
        reward = None
        action = None
        info = None
        
        # To let the first step run trough.
        done = False
        
        while True:
            # Store the latest env output in the prepared lists.
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            infos.append(info)
            
            # Abort if the episode terminated or max_steps
            # has been reached
            current_step += 1
            if done or current_step == max_steps:
                break            
            
            action = policy.get_action(observation=observation, reward=reward)
            
            if self.custom_frame_skip_length <= 1:
                observation, reward, done, info = self.env.step(action)
                if self.observation_callback is not None:
                    observation = self.observation_callback(observation)
                continue
            
            observations_step = []
            rewards_steps = []
            dones_stes = []
            infos_steps = []
            for i in range(self.custom_frame_skip_length):
                observation, reward, done, info = self.env.step(action)
                if self.observation_callback is not None:
                    observation = self.observation_callback(observation)
                observations_step.append(observation)
                rewards_steps.append(reward)
                dones_stes.append(done)
                infos_steps.append(info)
                
                if done:
                    break
            ordi = self.custom_frame_skip(observations=observations_step, 
                                          rewards=rewards_steps,
                                          dones=dones_stes,
                                          infos=infos_steps)
            observation, reward, done, info = ordi
            
        return observations, actions, rewards, infos

    def extract_episode_statistics(self,
                                   policy_class,
                                   n_episodes,
                                   start_seed=0,
                                   max_steps=-1,
                                   policy_kw_args=None,
                                   return_observations=False):
        """ 
        Compute episode information for many episodes
        
        E.g. return, number of frames until don,
                                   policy_kw_args={}e 

        Arguements:
        -----------
        policy_class: object
            The class of the policy. Will be initiated with a
            seed that is the episode number, counted from
            start_seed to n_episodes+start_seed.
        n_episodes: int
            How many episodes shall be produced
        start_seed: int
            The first seed, see also policy_class.
        policy: object
            Will be called as policy.get_action(observation, reward) to derivce
            the action for the next steps.
        max_step: int or None
            If int the maximum number of steps one episode should contain,
            including the initial state before the game starts.
            If -1 the episode will be run until it terminates i.e.
            env.step returns done=True
        policy_kw_args: dict or None
            Additional keyword arguments passed to policy_class init.
        return_observations: bool
            if True also returns the observations of the episodes

        Returns:
        --------
        episodes_df: Pandas Dataframe
            with seed as index and columns: [number of 
            steps until done or max_steps, total reward]
        all_observations: dict
            Only if return_observations.
            Keys are the seeds, values the lists of observed arrays.
        """
        start_time = dt.utcnow()
        number_of_steps = []
        total_reward = []
        seeds = []

        if policy_kw_args is None:
            policy_kw_args = {}

        if return_observations:
            all_observations = {}
        
        for random_seed in range(start_seed, n_episodes + start_seed):
            policy = policy_class(random_seed=random_seed, **policy_kw_args)
            episode_data = self.extract_episode(policy=policy,
                                                max_steps=max_steps)
            observations, actions, rewards, infos = episode_data

            if return_observations:
                all_observations[random_seed] = observations

            number_of_steps.append(len(observations))
            total_reward.append(sum(rewards[1:]))
            seeds.append(random_seed)

        df_data = {'total_reward': total_reward, 
                   'number_of_steps': number_of_steps}
        episode_df = pd.DataFrame(index=seeds, data=df_data)

        took_seconds = (dt.utcnow() - start_time).total_seconds()
        log.debug('Extracted {} episodes in {:.2f} seconds.'
                  .format(len(episode_df), took_seconds))

        if return_observations:
            return episode_df, all_observations
        else:
            return episode_df


    def extract_n_steps(self,
                        policy_class,
                        n_steps=10000,
                        start_seed=0,
                        max_steps=-1,
                        policy_kw_args=None):
        """
        Extract n steps from environmet and return as array.

        Arguements:
        -----------
        policy_class: object
            The class of the policy. Will be initiated with a
            seed that is the episode number, counted from
            start_seed to n_episodes+start_seed.
        n_steps: int
            The number of steps that should be extracted.
        start_seed: int
            The first seed, see also policy_class.
        policy: object
            Will be called as policy.get_action(observation, reward) to derivce
            the action for the next steps.
        max_step: int or None
            If int the maximum number of steps one episode should contain,
            including the initial state before the game starts.
            If -1 the episode will be run until it terminates i.e.
            env.step returns done=True
        policy_kw_args: dict or None
            Additional keyword arguments passed to policy_class init.

        Returns:
        --------
        n_observations: array
            steps at axis 0, other dims as returned by env.
        """
        start_time = dt.utcnow()

        # Quick exit for invalid number of steps.
        if n_steps <= 0:
            return np.array([])
        
        if policy_kw_args is None:
            policy_kw_args = {}

        # Extract episodes until enough steps have been recorded.
        random_seed = start_seed
        obs_arrays = []
        while True:
            policy = policy_class(random_seed=random_seed, **policy_kw_args)
            episode_data = self.extract_episode(policy=policy,
                                                max_steps=max_steps)
            observations, actions, rewards, infos = episode_data
            obs_array = np.stack(observations)
            obs_arrays.append(obs_array)

            random_seed += 1

            if sum([a.shape[0] for a in obs_arrays]) >= n_steps:
                break

        # Glue observations of episodes together and trim length
        # to desired number of steps.
        n_observations = np.concatenate(obs_arrays)[:n_steps]

        took_seconds = (dt.utcnow() - start_time).total_seconds()
        log.info('Extracted steps from {} episodes in {:.2f} seconds.'
              .format(random_seed-start_seed, took_seconds))

        return n_observations

    @staticmethod
    def create_black_boundary(observations, black_bound_shape):
        """
        Create a black boundary around the observations.

        The observations will be placed central in the
        black boundary.

        Arguments:
        ----------
        observations: array or list
            Must be list of arrays with shape: (y, x, channel) or
            stacked version of such list with (step, y, x, channel)

        black_bound_shape: tuple of int
            as (y_new, x_new) the new dimension of y and x.
            y_new, x_new must be larger then y, x

        Returns:
        --------
        obs_with_bounds: list of arrays
            list of arrays with shape: (y, x, channel)
        """
        y_new, x_new = black_bound_shape

        obs_with_bounds = []
        for obs_array in observations:
            y, x, channel = obs_array.shape
            new_dim = (y_new, x_new, channel)

            # Create an array with all black pixels
            obs_array_wb = np.zeros(new_dim, dtype=obs_array.dtype)

            # Compute the index of the right upper corner
            # of the observations image in the image with
            # black boundary
            i_x = int(np.floor((x_new-x)/2))
            i_y = int(np.floor((y_new-y)/2))

            # Place the obs image within the black bound.
            obs_array_wb[i_y:i_y+y, i_x:i_x+x, :] = obs_array[:]
            obs_with_bounds.append(obs_array_wb)

        return obs_with_bounds


    def __del__(self):
        """
        Tidy up on exit, altough it shouldn't be strictly necessary.
        """
        self.env.close()


class EpochObsProvider(object):

    def __init__(self, policy_class=RandomPolicy, env_name='SpaceInvaders-v4',
                     custom_frame_skip_length=1, observation_callback=None,
                     n_processes=4, n_queued_obs=1, n_steps=10000, start_seed=0,
                     max_steps=-1, black_bound_shape=(256, 256)):
        """
        This class may be used to compute the obs aka the environment
        obervations in a asynchronous way. Therefore a a number of
        n_queued_obs observations will be computed in parallel.
        Once one observation is poped, the programm will compute
        a new observations object, thus leading to a state where
        always n_queued_obs observations are under computation or
        finished.

        Arguments:
        ----------
        policy_class: object
            The class of the policy. Will be initiated with a
            seed that is the episode number, counted from
            start_seed to n_episodes+start_seed.
        env_name: String
            The name of the environment to use. As expected by gym.make
        use_custom_frame_skip: bool
            Applies self.custom_frame_skip function to custom_frame_skip_length
            outputs of env.step while repeating the action. No effect for 
            custom_frame_skip_length = 1.
        observation_callback: None or fuction
            If not None: Will be called as observation_callback(observation)
            directly after env.step returns the observations. Will be applied
            before custom_frame_skip functions gets into action.
        n_processes: int
            How many parallel processes to use for computation.
        n_queued_obs: int
            see above
        n_steps: int
            The number of steps that should be extracted.
        start_seed: int
            The first seed, see also policy_class.
        max_step: int or None
            If int the maximum number of steps one episode should contain,
            including the initial state before the game starts.
            If -1 the episode will be run until it terminates i.e.
            env.step returns done=True
        black_bound_shape: tuple of int
            as in ExtractWorker.create_black_boundary
        """
        self._pool = Pool(n_processes)
        self._obs_jobs = []

        # Store all arguments for _start_obs_computation() to use
        self.policy_class = policy_class
        self.env_name = env_name
        self.custom_frame_skip_length = custom_frame_skip_length
        self.observation_callback = observation_callback
        self.n_steps = n_steps
        self.seed = start_seed
        self.max_steps = max_steps
        self.black_bound_shape = black_bound_shape

        for _ in range(n_queued_obs):
            self._start_obs_computation()


    @staticmethod
    def _pool_worker(job):
        """
        Do the actual computation of the observations.

        Arguments:
        ----------
        job: dict
            Arguments for the process, see code.

        Returns:
        --------
        epoch_obs: array
            Observations with boundaries
        """
        ew = ExtractWorker(env_name=job['env_name'],
                           custom_frame_skip_length=job['custom_frame_skip_length'],
                           observation_callback=job['observation_callback'])

        epoch_obs = ew.extract_n_steps(policy_class=job['policy_class'],
                                       n_steps=job['n_steps'],
                                       start_seed=job['start_seed'],
                                       max_steps=job['max_steps'])
        epoch_obs = ew.create_black_boundary(epoch_obs,
                                             black_bound_shape=job['black_bound_shape'])
        epoch_obs = np.stack(epoch_obs)
        return epoch_obs


    def _start_obs_computation(self):
        """
        Start the computation of a new observations object.
        """
        job = {'env_name': self.env_name,
               'custom_frame_skip_length': self.custom_frame_skip_length,
               'observation_callback': self.observation_callback,
               'policy_class': self.policy_class,
               'n_steps': self.n_steps,
               'start_seed': self.seed,
               'max_steps': self.max_steps,
               'black_bound_shape': self.black_bound_shape,
              'test': 'test'}
        obs_job = self._pool.apply_async(self._pool_worker, [job])
        self._obs_jobs.append(obs_job)
        self.seed += 1


    def pop_observations(self):
        """
        Return one object of observations

        Returns:
        --------
        epoch_obs: array
            Observations with boundaries
        """
        log.info('Fetching observations from queue')
        # Take one from the queue
        first_obs_job = self._obs_jobs.pop(0)

        # Retrieve the result of the wworker
        start_time = dt.now()
        epoch_obs = first_obs_job.get()
        end_time = dt.now()
        get_time = np.round((end_time - start_time).total_seconds(), 2)
        log.info('Waited {:.02}s on observations to compute'
                 .format(get_time))

        # Start a new one to keep the queue in balance.
        # Do it here as you will else have the situation
        # at where you are computing one observations object
        # more then you set n_queued_obs.
        self._start_obs_computation()

        return epoch_obs
