import gym
environment = gym.make('FrozenLake-v0')
environment.reset()
environment.render()
import time

for dummy in range(100):
    time.sleep(.1)
    environment.render()
    environment.step(environment.action_space.sample())

time.sleep(10)

