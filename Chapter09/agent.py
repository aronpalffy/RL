from actor import Actor
from critic import Critic

import numpy as np
from numpy.random import choice
import random
from collections import namedtuple, deque


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
    
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self, batch_size=32):
        return random.sample(self.memory, k=self.batch_size)
    
    def __len__(self):
        return len(self.memory)
    
    
class Agent:
    def __init__(self, state_size, batch_size, is_eval = False):
        self.state_size = state_size
        self.action_size = 3
        self.buffer_size = 1000000
        self.batch_size = batch_size
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)
        self.inventory = []
        self.is_eval = is_eval
        
        self.gamma = 0.99  # discount factor
        self.tau = 0.001  # for soft update of target parameters

        self.epsilon = 1.0
        self.epsion_min = 0.01
        """
        self.epsilon_decay = 0.999999 with this setting
        epsilon decays to 0.5057234296206272 in 301 epizodes of training
        on orig_training_14-08-2006-13-08-2015^GSPC 
        """
        """
        self.epsilon_decay = 0.99999 with this setting
        epsilon decays to 0.009999971601096055 in 301 epizodes of training
        on orig_training_14-08-2006-13-08-2015^GSPC 
        """
        self.epsilon_decay = 0.999995
        
        self.actor_local = Actor(self.state_size, self.action_size)
        self.actor_target = Actor(self.state_size, self.action_size)    

        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)
        
        self.critic_target.model.set_weights(self.critic_local.model.get_weights()) 
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())
        
    def act(self, state):
        options = self.actor_local.model.predict(state)
        self.last_state = state
        if not self.is_eval and random.random() <= self.epsilon:
            return choice(range(3), p = options[0])
        return np.argmax(options[0])
    
    def step(self, action, reward, next_state, done):
        self.memory.add(self.last_state, action, reward, next_state, done)
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)
            self.last_state = next_state
        if self.epsilon > self.epsion_min:
            self.epsilon*=self.epsilon_decay

    def learn(self, experiences):               
        states = np.vstack([e.state for e in experiences if e is not None]).astype(np.float32).reshape(-1,self.state_size)    
        actions = np.vstack([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1,self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1,1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.float32).reshape(-1,1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None]).astype(np.float32).reshape(-1,self.state_size)

        # update critic
        ## get predicted next-state actions and Q values from target models
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])
        # compute Q targets for current states
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        
        self.critic_local.model.train_on_batch(x = [states, actions], y=Q_targets)
        
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]),(-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])
 
        # update target networks
        self.soft_update(self.critic_local.model, self.critic_target.model)  
        self.soft_update(self.actor_local.model, self.actor_target.model)

    def soft_update(self, local_model, target_model):
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights)

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)
