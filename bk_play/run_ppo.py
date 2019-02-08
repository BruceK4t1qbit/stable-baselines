import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
# from stable_baselines import PPO2
from stable_baselines import SAC
from stable_baselines.sac import policies

import roboschool

# env = gym.make('CartPole-v1')
# env = gym.make('MountainCar-v0')
# env = gym.make('Hopper-v2')

# env = gym.make("RoboschoolInvertedPendulum-v1")
# env = gym.make("RoboschoolInvertedPendulumSwingup-v1")
# env = gym.make("RoboschoolInvertedDoublePendulum-v1")
# env = gym.make("RoboschoolReacher-v1")
env = gym.make("RoboschoolHopper-v1")
# env = gym.make("RoboschoolWalker2d-v1")
# env = gym.make("RoboschoolHalfCheetah-v1")
# env = gym.make("RoboschoolAnt-v1")
# env = gym.make("RoboschoolHumanoid-v1")
# env = gym.make("RoboschoolHumanoidFlagrun-v1")
# env = gym.make("RoboschoolAtlasForwardWalk-v1")
# env = gym.make("RoboschoolPong-v1")

env = VecNormalize(DummyVecEnv([lambda: env]))  # The algorithms require a vectorized environment to run


# model = PPO2(MlpPolicy, env, verbose=1)
model = SAC(policies.FeedForwardPolicy,
            env,
            verbose=1,
            policy_kwargs=dict(feature_extraction="mlp"),
            buffer_size=10000,
            gradient_steps=5,
            learning_rate=3e-5)  # todo try lowering the target entropy!  Also, try lowering the learning rate b/c maybe hitting edge cases too rarely!
model.learn(total_timesteps=100000)

obs = env.reset()
n = 1000
for i in range(n):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
print("done")