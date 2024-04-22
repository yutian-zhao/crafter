import argparse

import crafter
import stable_baselines3
import gymnasium as gym

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', default='logdir/crafter_reward-ppo/0')
parser.add_argument('--steps', type=float, default=1e6)
args = parser.parse_args()

env = crafter.Env()
# env = gym.make('CrafterReward-v1')
env = crafter.Recorder(
    env, args.outdir,
    save_stats=True,
    save_episode=False,
    save_video=False,
)
env = gym.wrappers.StepAPICompatibility(env)

model = stable_baselines3.PPO('CnnPolicy', env, verbose=1)
model.learn(total_timesteps=args.steps)
