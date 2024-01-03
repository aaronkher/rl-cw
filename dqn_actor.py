# The constants that are defined in config.json:
import tmrl.config.config_constants as cfg
# Useful classes:
import tmrl.config.config_objects as cfg_obj
# The utility that TMRL uses to partially instantiate classes:
from tmrl.util import partial
# The TMRL three main entities (i.e., the Trainer, the RolloutWorker and the central Server):
from tmrl.networking import Trainer, RolloutWorker, Server

# The training class that we will customize with our own training algorithm in this tutorial:
from tmrl.training_offline import TrainingOffline

# And a couple external libraries:
import numpy as np
import os

# Maximum number of training 'epochs':
# (training is checkpointed at the end of each 'epoch', this is also when training metrics can be logged to wandb)
epochs = cfg.TMRL_CONFIG["MAX_EPOCHS"]

# Number of rounds per 'epoch':
# (training metrics are displayed in the terminal at the end of each round)
rounds = cfg.TMRL_CONFIG["ROUNDS_PER_EPOCH"]

# Number of training steps per round:
# (a training step is a call to the train() function that we will define later in this tutorial)
steps = cfg.TMRL_CONFIG["TRAINING_STEPS_PER_ROUND"]

# Minimum number of environment steps collected before training starts:
# (this is useful when you want to fill your replay buffer with samples from a baseline policy)
start_training = cfg.TMRL_CONFIG["ENVIRONMENT_STEPS_BEFORE_TRAINING"]

# Maximum training steps / environment steps ratio:
# (if training becomes faster than this ratio, it will be paused, waiting for new samples from the environment)
max_training_steps_per_env_step = cfg.TMRL_CONFIG["MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP"]

# Number of training steps performed between broadcasts of policy updates:
update_model_interval = cfg.TMRL_CONFIG["UPDATE_MODEL_INTERVAL"]

# Number of training steps performed between retrievals of received samples to put them in the replay buffer:
update_buffer_interval = cfg.TMRL_CONFIG["UPDATE_BUFFER_INTERVAL"]

# Training device (e.g., "cuda:0"):
device_trainer = 'cuda' if cfg.CUDA_TRAINING else 'cpu'

# Maximum size of the replay buffer:
memory_size = cfg.TMRL_CONFIG["MEMORY_SIZE"]

# Batch size for training:
batch_size = cfg.TMRL_CONFIG["BATCH_SIZE"]

# Wandb credentials:
# (Change this with your own if you want to keep your training curves private)
# (Also, please use your own wandb account if you are going to log huge stuff :) )

wandb_run_id = cfg.WANDB_RUN_ID  # change this by a name of your choice for your run
wandb_project = cfg.TMRL_CONFIG["WANDB_PROJECT"]  # name of the wandb project in which your run will appear
wandb_entity = cfg.TMRL_CONFIG["WANDB_ENTITY"]  # wandb account
wandb_key = cfg.TMRL_CONFIG["WANDB_KEY"]  # wandb API key

os.environ['WANDB_API_KEY'] = wandb_key  # this line sets your wandb API key as the active key

# Number of time-steps after which episodes collected by the worker are truncated:
max_samples_per_episode = cfg.TMRL_CONFIG["RW_MAX_SAMPLES_PER_EPISODE"]

# Networking parameters:
# (In TMRL, networking is managed by tlspyo. The following are tlspyo parameters.)
server_ip_for_trainer = cfg.SERVER_IP_FOR_TRAINER  # IP of the machine running the Server (trainer point of view)
server_ip_for_worker = cfg.SERVER_IP_FOR_WORKER  # IP of the machine running the Server (worker point of view)
server_port = cfg.PORT  # port used to communicate with this machine
password = cfg.PASSWORD  # password that secures your communication
security = cfg.SECURITY  # when training over the Internet, it is safer to change this to "TLS"
# (please read the security instructions on GitHub)

# =====================================================================
# ADVANCED PARAMETERS
# =====================================================================
# You may want to change the following in advanced applications;
# however, most competitors will not need to change this.
# If interested, read the full TMRL tutorial on GitHub.
# These parameters are to change here directly (if you want).
# (Note: The tutorial may stop working if you change these)

# Base class of the replay memory used by the trainer:
memory_base_cls = cfg_obj.MEM

# Sample compression scheme applied by the worker for this replay memory:
sample_compressor = cfg_obj.SAMPLE_COMPRESSOR

# Sample preprocessor for data augmentation:
sample_preprocessor = None

# Path from where an offline dataset can be loaded to initialize the replay memory:
dataset_path = cfg.DATASET_PATH

# Preprocessor applied by the worker to the observations it collects:
# (Note: if your script defines the name "obs_preprocessor", we will use your preprocessor instead of the default)
obs_preprocessor = cfg_obj.OBS_PREPROCESSOR


# =====================================================================
# COMPETITION FIXED PARAMETERS
# =====================================================================
# Competitors CANNOT change the following parameters.

# rtgym environment class (full TrackMania Gymnasium environment):
env_cls = cfg_obj.ENV_CLS

# Device used for inference on workers (change if you like but keep in mind that the competition evaluation is on CPU)
device_worker = 'cpu'


# =====================================================================
# ENVIRONMENT PARAMETERS
# =====================================================================
# You are allowed to customize these environment parameters.
# Do not change these here though, customize them in config.json.
# Your environment configuration must be part of your submission,
# e.g., the "ENV" entry of your config.json file.

# Dimensions of the TrackMania window:
window_width = cfg.WINDOW_WIDTH  # must be between 256 and 958
window_height = cfg.WINDOW_HEIGHT  # must be between 128 and 488

# Dimensions of the actual images in observations:
img_width = cfg.IMG_WIDTH
img_height = cfg.IMG_HEIGHT

# Whether you are using grayscale (default) or color images:
# (Note: The tutorial will stop working if you use colors)
img_grayscale = cfg.GRAYSCALE

# Number of consecutive screenshots in each observation:
imgs_buf_len = cfg.IMG_HIST_LEN

# Number of actions in the action buffer (this is part of observations):
# (Note: The tutorial will stop working if you change this)
act_buf_len = cfg.ACT_BUF_LEN


# =====================================================================
# MEMORY CLASS
# =====================================================================
# Nothing to do here.
# This is the memory class passed to the Trainer.
# If you need a custom memory, change the relevant advanced parameters.
# Custom memories are described in the full TMRL tutorial.

memory_cls = partial(memory_base_cls,
                     memory_size=memory_size,
                     batch_size=batch_size,
                     sample_preprocessor=sample_preprocessor,
                     dataset_path=cfg.DATASET_PATH,
                     imgs_obs=imgs_buf_len,
                     act_buf_len=act_buf_len,
                     crc_debug=False)


# =====================================================================
# CUSTOM MODEL
# =====================================================================
# Alright, now for the fun part.
# Our goal in this competition is to come up with the best trained
# ActorModule for TrackMania 2020, where an 'ActorModule' is a policy.
# In this tutorial, we present a deep RL way of tackling this problem:
# we implement our own deep neural network architecture (ActorModule),
# and then we implement our own RL algorithm to train this module.

# We will implement SAC and a hybrid CNN/MLP model.

# The following constants are from the Spinup implementation of SAC
# that we simply copy/paste and adapt in this tutorial.
LOG_STD_MAX = 2
LOG_STD_MIN = -20


# Let us import the ActorModule that we are supposed to implement.
# We will use PyTorch in this tutorial.
# TMRL readily provides a PyTorch-specific subclass of ActorModule:
from tmrl.actor import TorchActorModule

# Plus a couple useful imports:
import torch
import torch.nn as nn
import torch.nn.functional as F
from trackmania_dqn.dqn import TrackManiaDQN
from torch.distributions.normal import Normal
from math import floor


# We can now implement the TMRL ActorModule interface that we are supposed to submit for this competition.

# During training, TMRL will regularly save our trained ActorModule in the TmrlData/weights folder.
# By default, this would be done using the torch (i.e., pickle) serializer.
# However, while saving and loading your own pickle files is fine,
# it is highly dangerous to load other people's pickle files.
# Therefore, the competition submission does not accept pickle files.
# Instead, we can submit our trained weights in the form of a human-readable JSON file.
# The ActorModule interface defines save() and load() methods that we will override with our own JSON serializer.

import json


class TorchJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for torch tensors, used in the custom save() method of our ActorModule.
    """
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        return json.JSONEncoder.default(self, obj)


class TorchJSONDecoder(json.JSONDecoder):
    """
    Custom JSON decoder for torch tensors, used in the custom load() method of our ActorModule.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, dct):
        for key in dct.keys():
            if isinstance(dct[key], list):
                dct[key] = torch.Tensor(dct[key])
        return dct


class MyActorModule(TorchActorModule):
    """
    Our policy wrapped in the TMRL ActorModule class.

    The only required method is ActorModule.act().
    We also implement a forward() method for our training algorithm.

    (Note: TorchActorModule is a subclass of ActorModule and torch.nn.Module)
    """
    def __init__(self, observation_space, action_space):
        """
        When implementing __init__, we need to take the observation_space and action_space arguments.

        Args:
            observation_space: observation space of the Gymnasium environment
            action_space: action space of the Gymnasium environment
        """
        # We must call the superclass __init__:
        super().__init__(observation_space, action_space)
        self.DQN = TrackManiaDQN(
            observation_space, 
            action_space,
            gamma=0.99,
            epsilon_start=0.9,
            epsilon_min=0.05,
            epsilon_decay=0.02,
        )


    def save(self, path):
        """
        JSON-serialize a detached copy of the ActorModule and save it in path.

        IMPORTANT: FOR THE COMPETITION, WE ONLY ACCEPT JSON AND PYTHON FILES.
        IN PARTICULAR, WE *DO NOT* ACCEPT PICKLE FILES (such as output by torch.save()...).

        All your submitted files must be human-readable, for everyone's safety.
        Indeed, untrusted pickle files are an open door for hackers.

        Args:
            path: pathlib.Path: path to where the object will be stored.
        """
        with open(path, 'w') as json_file:
            json.dump(self.state_dict(), json_file, cls=TorchJSONEncoder)
        # torch.save(self.state_dict(), path)

    def load(self, path, device):
        """
        Load the parameters of your trained ActorModule from a JSON file.

        Adapt this method to your submission so that we can load your trained ActorModule.

        Args:
            path: pathlib.Path: full path of the JSON file
            device: str: device on which the ActorModule should live (e.g., "cpu")

        Returns:
            The loaded ActorModule instance
        """
        self.device = device
        with open(path, 'r') as json_file:
            state_dict = json.load(json_file, cls=TorchJSONDecoder)
        self.load_state_dict(state_dict)
        self.to_device(device)
        # self.load_state_dict(torch.load(path, map_location=self.device))
        return self

    def forward(self, obs, test=False, compute_logprob=True):
        """
        Computes the output action of our policy from the input observation.

        The whole point of deep RL is to train our policy network (actor) such that it outputs relevant actions.
        Training per-se will also rely on a critic network, but this is not part of the trained policy.
        Thus, our ActorModule will only implement the actor.

        Args:
            obs: the observation from the Gymnasium environment (when using TorchActorModule this is a torch.Tensor)
            test (bool): this is True for test episodes (deployment) and False for training episodes;
                in SAC, this enables us to sample randomly during training and deterministically at test-time.
            compute_logprob (bool): SAC will set this to True to retrieve log probabilities.

        Returns:
            the action sampled from our policy from observation obs
            the log probability of this action (this will be used for SAC)
        """
        # obs is our input observation.
        # We feed it to our actor neural network, which will output an action.
        best_action = self.DQN.get_best_action(obs)
        return best_action




    # Now, the only method that all participants are required to implement is act()
    # act() is the interface for TMRL to use your ActorModule as the policy it tests in TrackMania.
    # For the evaluation, the "test" argument will be set to True.
    def act(self, obs, test=False):
        """
        Computes an action from an observation.

        This method is the one all participants must implement.
        It is the policy that TMRL will use in TrackMania to evaluate your submission.

        Args:
            obs (object): the input observation (when using TorchActorModule, this is a torch.Tensor)
            test (bool): True at test-time (e.g., during evaluation...), False otherwise

        Returns:
            act (numpy.array): the computed action, in the form of a numpy array of 3 values between -1.0 and 1.0
        """
        # Since we have already implemented our policy in the form of a neural network,
        # act() is now pretty straightforward.
        # We don't need to compute the log probabilities here (they will be for our SAC training algorithm).
        # Also note that, when using TorchActorModule, TMRL calls act() in a torch.no_grad() context.
        # Thus, you don't need to use "with torch.no_grad()" here.
        # But let us do it anyway to be extra sure, for the people using ActorModule instead of TorchActorModule.
        self.forward(obs)

        # TODO CONVERT TO NUMPY ARRAY IN FORM OF ACTION - still not 100% sure what this should be for discrete

        # Need to set "VIRTUAL_GAMEPAD": false in config.json in order to use 4 discrete keyboard inputs
        

from tmrl.training import TrainingAgent

# A TrainingAgent must implement two methods:
# -> train(batch): optimizes the model from a batch of RL samples
# -> get_actor(): outputs a copy of the current ActorModule
# In this tutorial, we implement the Soft Actor-Critic algorithm
# by adapting the OpenAI Spinup implementation.

class DQNTrainingAgent(TrainingAgent):
    """
    Our custom training algorithm (SAC in this tutorial).

    Custom TrainingAgents implement two methods: train(batch) and get_actor().
    The train method performs a training step.
    The get_actor method retrieves your ActorModule to save it and send it to the RolloutWorkers.

    Your implementation must also pass three required arguments to the superclass:

    - observation_space (gymnasium.spaces.Space): observation space (here for your convenience)
    - action_space (gymnasium.spaces.Space): action space (here for your convenience)
    - device (str): device that should be used for training (e.g., `"cpu"` or `"cuda:0"`)
    """
    # no-grad copy of the model used to send the Actor weights in get_actor():
    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __init__(self,
                 observation_space=None,  # Gymnasium observation space (required argument here for your convenience)
                 action_space=None,  # Gymnasium action space (required argument here for your convenience)
                 device=None,  # Device our TrainingAgent should use for training (required argument)
                 model_cls=TrackManiaDQN,  # An actor-critic module, encapsulating our ActorModule
                 ):

        # required arguments passed to the superclass:
        super().__init__(observation_space=observation_space,
                         action_space=action_space,
                         device=device)
        
        self.model_cls = model_cls

    def get_actor(self):
        """
        Returns a copy of the current ActorModule.

        We return a copy without gradients, as this is for sending to the RolloutWorkers.

        Returns:
            actor: ActorModule: updated actor module to forward to the worker(s)
        """
        return self.model_nograd.actor

    def train(self, batch):
        """
        Executes a training iteration from batched training samples (batches of RL transitions).

        A training sample is of the form (o, a, r, o2, d, t) where:
        -> o is the initial observation of the transition
        -> a is the selected action during the transition
        -> r is the reward of the transition
        -> o2 is the final observation of the transition
        -> d is the "terminated" signal indicating whether o2 is a terminal state
        -> t is the "truncated" signal indicating whether the episode has been truncated by a time-limit

        Note that we will ignore the t signal for a clever reason.
        When the episode is truncated due to a time limit, we do not want our model to believe that o2 is terminal.
        Instead, we will make our model believe that the episode would normally have continued after this transition.
        Because we use a discount factor, this does not result in an infinite value.
        In our situation, the discount factor is what incentivizes the AI to run as fast as possible!

        Args:
            batch: (previous observation, action, reward, new observation, terminated signal, truncated signal)

        Returns:
            logs: Dictionary: a python dictionary of training metrics you wish to log on wandb
        """
        self.model_cls.train(batch)