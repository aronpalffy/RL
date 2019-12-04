import gym
import time
environment = gym.make('SpaceInvaders-v0')
environment.reset()
environment.render()

for dummy in range(1000):
    time.sleep(.001)
    environment.render()
    environment.step(environment.action_space.sample())

print("done")
#time.sleep(10)