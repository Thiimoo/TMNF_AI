import gym
from stable_baselines3 import DQN
from TrackmaniaAI.trackmania_env import TrackmaniaEnv

# Create and register the custom environment
env = TrackmaniaEnv()

# Load the previously trained model if it exists; otherwise, create a new one
try:
    model = DQN.load("trackmania_dqn", env=env)
    print("Model loaded. Continuing training.")
except FileNotFoundError:
    model = DQN("CnnPolicy", env, learning_rate=0.0001, verbose=1)
    print("No existing model found. Training a new model.")

# Train the model
model.learn(total_timesteps=200)

# Save the model after training
model.save("trackmania_dqn")
