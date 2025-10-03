import torch as th
import onnxruntime as ort
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3 import PPO

from env import DotsAndBoxesEnv
import gymnasium as gym
import numpy as np

class DotsAndBoxesSingleAgentWrapper(gym.Env):
    def __init__(self, env_class, grid_size=5, agent_name="player_1"):
        super().__init__()
        self.env = env_class(grid_size=grid_size)
        self.agent_name = agent_name

        self.observation_space = self.env.observation_spaces[self.agent_name]
        self.action_space = self.env.action_spaces[self.agent_name]

    def reset(self, seed=None, options=None):
        obs_dict = self.env.reset(seed=seed)
        obs = obs_dict[self.agent_name]
        return obs, {}

    def step(self, action):
        self.env.step(action)

        obs = self.env.observe(self.agent_name)
        reward = self.env.rewards[self.agent_name]
        terminated = self.env.terminations[self.agent_name]
        truncated = self.env.truncations[self.agent_name]
        info = self.env.infos[self.agent_name]

        return obs, reward, terminated, truncated, info

    def render(self):
        self.env.render("human")

env = DotsAndBoxesSingleAgentWrapper(DotsAndBoxesEnv, grid_size=5)

class OnnxableSB3Policy(th.nn.Module):
    def __init__(self, policy: BasePolicy):
        super().__init__()
        self.policy = policy

    def forward(self, observation: th.Tensor):
        with th.no_grad():
            distribution = self.policy.get_distribution(observation)
            
            if hasattr(distribution, 'distribution'):
                logits = distribution.distribution.logits
            elif hasattr(distribution, 'logits'):
                logits = distribution.logits
            else:
                features = self.policy.extract_features(observation)
                latent_pi = self.policy.mlp_extractor.forward_actor(features)
                logits = self.policy.action_net(latent_pi)
            
        return logits

model = PPO.load("models/dots_and_boxes_model")

onnx_policy = OnnxableSB3Policy(model.policy)
obs, _ = env.reset()
obs_tensor = obs_as_tensor(obs, model.policy.device).unsqueeze(0)

th.onnx.export(
    onnx_policy,
    args=(obs_tensor,),
    f="models/dots_and_boxes_model.onnx",
    input_names=["obs"],
    output_names=["action_logits"],
    opset_version=17,
    do_constant_folding=True,
)

print("Model exported successfully!")