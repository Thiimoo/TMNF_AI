import gym
from stable_baselines3 import DQN
from trackmania_env import TrackmaniaEnv

# Create and register the custom environment
env = TrackmaniaEnv()

# Define the RL model (using DQN)
model = DQN("CnnPolicy", env, learning_rate=0.0001, verbose=1)

# Train the model
model.learn(total_timesteps=10000)

# Save the model
model.save("trackmania_dqn")

# To load and evaluate the model:
# model = DQN.load("trackmania_dqn", env=env)
# obs = env.reset()
# while True:
#     action, _ = model.predict(obs)
#     obs, reward, done, _ = env.step(action)
#     if done:
#         obs = env.reset()
