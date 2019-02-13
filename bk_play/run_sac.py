
from numpy import array, ndarray
import numpy as np
import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
# from stable_baselines import PPO2
from stable_baselines import SAC
from stable_baselines.sac import policies

import roboschool


#  __t:  Environments and Config Parameters
learning_rate=3e-4

# env = gym.make('CartPole-v1')
# env = gym.make('MountainCar-v0')
# env = gym.make('Hopper-v2')

# env = gym.make("RoboschoolInvertedPendulum-v1")
# env = gym.make("RoboschoolInvertedPendulumSwingup-v1")
# env = gym.make("RoboschoolInvertedDoublePendulum-v1")
# env = gym.make("RoboschoolReacher-v1")
# env = gym.make("RoboschoolHopper-v1")
# env = gym.make("RoboschoolWalker2d-v1")
# env = gym.make("RoboschoolHalfCheetah-v1")

buffer_size=1000000
gradient_steps=1
env = gym.make("RoboschoolAnt-v1")
target_entropy=-np.prod(env.action_space.shape).astype(np.float32)*.75

# buffer_size=200000
# gradient_steps=5
# env = gym.make("RoboschoolHumanoid-v1")
# target_entropy=-np.prod(env.action_space.shape).astype(np.float32)*1

# env = gym.make("RoboschoolHumanoidFlagrun-v1")
# env = gym.make("RoboschoolAtlasForwardWalk-v1")
# env = gym.make("RoboschoolPong-v1")


# env = VecNormalize(DummyVecEnv([lambda: env]))  # The algorithms require a vectorized environment to run

# The algorithms require a vectorized environment to run
# The SAC algorithm doesn't use normalization in the rl-zoo!
env = DummyVecEnv([lambda: env])

from stable_baselines.sac.policies import FeedForwardPolicy as SACPolicy
from stable_baselines.common.policies import register_policy

class CustomSACPolicy(SACPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomSACPolicy, self).__init__(*args, **kwargs,
                                              layers=[256, 256],
                                              feature_extraction="mlp")
register_policy('CustomSACPolicy', CustomSACPolicy)  #  __c:  Could probably just provide this directly as well!

hyper_params = {'policy': 'CustomSACPolicy',  #  __c:  It will figure out the
 'learning_rate': lambda f : f*3e-4,  # f is the fraction left to go
 'buffer_size': 1000000,
 'batch_size': 256,
 'ent_coef': 0.01,
 'train_freq': 1,
 'gradient_steps': 1,
 'learning_starts': 1000}

# model = PPO2(MlpPolicy, env, verbose=1)
model = SAC(env=env,
            verbose=1,
            **hyper_params
            )
# model = SAC('CustomSACPolicy',
#             env,
#             verbose=1,
#             policy_kwargs=dict(feature_extraction="mlp"),
#             buffer_size=buffer_size,
#             gradient_steps=gradient_steps,
#             learning_rate=learning_rate,
#             target_entropy=-target_entropy
#             )  # todo try lowering the target entropy!  Also, try lowering the learning rate b/c maybe hitting edge cases too rarely!
model.learn(total_timesteps=6000000)

obs = env.reset()
n = 1000
for i in range(n):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
print("done")