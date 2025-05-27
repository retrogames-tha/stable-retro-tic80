"""
Train an agent using Proximal Policy Optimization from Stable Baselines 3
"""

import argparse
from datetime import datetime
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
)
from pathlib import Path
import retro


class StochasticFrameSkip(gym.Wrapper):
    def __init__(self, env, n, stickprob):
        gym.Wrapper.__init__(self, env)
        self.n = n
        self.stickprob = stickprob
        self.curac = None
        self.rng = np.random.RandomState()
        self.supports_want_render = hasattr(env, "supports_want_render")

    def reset(self, **kwargs):
        self.curac = None
        return self.env.reset(**kwargs)

    def step(self, ac):
        terminated = False
        truncated = False
        totrew = 0
        for i in range(self.n):
            if(terminated or truncated):
                print(terminated or truncated)
            
            
            # First step after reset, use action
            if self.curac is None:
                self.curac = ac
            # First substep, delay with probability=stickprob
            elif i == 0:
                if self.rng.rand() > self.stickprob:
                    self.curac = ac
            # Second substep, new action definitely kicks in
            elif i == 1:
                self.curac = ac
            if self.supports_want_render and i < self.n - 1:
                ob, rew, terminated, truncated, info = self.env.step(
                    self.curac,
                    want_render=False,
                )
                
            else:
                ob, rew, terminated, truncated, info = self.env.step(self.curac)
            totrew += rew
            #print(info)
            if terminated or truncated:
                break
            
        return ob, totrew, terminated, truncated, info


def make_retro(*, game, state=None, max_episode_steps=4500, **kwargs):
    if state is None:
        state = retro.State.DEFAULT
    env = retro.make(game, state, **kwargs)
    env = StochasticFrameSkip(env, n=4, stickprob=0.25)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def wrap_deepmind_retro(env):
    """
    Configure environment for retro games, using config similar to DeepMind-style Atari in openai/baseline's wrap_deepmind
    """
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    return env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="Arcade-Bomber-Tic-80")
    parser.add_argument("--state", default=retro.State.DEFAULT)
    parser.add_argument("--scenario", default=None)
    args = parser.parse_args()

    def make_env():
        env = make_retro(game=args.game, state=args.state, scenario=args.scenario)
        env = wrap_deepmind_retro(env)
        return env
    
    tensorboard_log_dir = "./logs"

    venv = VecTransposeImage(VecFrameStack(SubprocVecEnv([make_env] * 8), n_stack=4))
    model = PPO(
        policy="CnnPolicy",
        env=venv,
        #learning_rate=lambda f: f * 2.5e-4,
        #n_steps=128,
        #batch_size=32,
        #n_epochs=4,
        #gamma=0.99,
        #gae_lambda=0.95,
        #clip_range=0.1,
        #ent_coef=0.01,
        device="auto",
        verbose=1,
        tensorboard_log=tensorboard_log_dir,
        
    )
    iterations=0
    try:
        while True:
            if Path('./models/ppo_model.zip').is_file():
                model = PPO.load("./models/ppo_model", env=venv, print_system_info=True)
            model.tensorboard_log = tensorboard_log_dir
            model.learn(
                total_timesteps=4_000_000,
                log_interval=1,
                tb_log_name = f"PPO_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )
            model.save("./models/ppo_model")
            iterations+=1
    finally:
        model.save("./models/ppo_model")
        print(f"iterations: {iterations}")



if __name__ == "__main__":
    main()




