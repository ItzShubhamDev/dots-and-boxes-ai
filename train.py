import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from pettingzoo.utils import turn_based_aec_to_parallel
import torch
import numpy as np

from env import DotsAndBoxesEnv

def create_env(grid_size=5):
    env = DotsAndBoxesEnv(grid_size=grid_size)
    env = ss.pettingzoo_env_to_vec_env_v1(turn_based_aec_to_parallel(env))
    env = ss.concat_vec_envs_v1(env, num_vec_envs=8, base_class="stable_baselines3")
    return env

env = create_env()

# Exploration to understand basic game rules
model = PPO(
  MlpPolicy,
    env,
    learning_rate=3e-4,
    n_steps=4096,
    batch_size=512,
    n_epochs=15,
    gamma=0.95,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.05,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(
        net_arch=[512, 512, 256],
        activation_fn=torch.nn.ReLU
    ),
    verbose=1,
)

model.learn(total_timesteps=800_000, progress_bar=True)

# Reduce exploration and focus on strategy refinement
model.learning_rate = 1e-4
model.ent_coef = 0.02
model.gamma = 0.98

model.learn(total_timesteps=1_000_000, progress_bar=True)

# Fine tuning with higher stability and lower exploration
model.learning_rate = 3e-5
model.ent_coef = 0.005
model.gamma = 0.99

model.learn(total_timesteps=500_000, progress_bar=True)
model.save("models/dots_and_boxes_model")

print("Training completed")