import random
import numpy as np
import time

class Replay2:
    def __init__(self, batchSize):
        self.batchSize = batchSize
        self.size = 10000
        self.samples = []

    def clear(self):
        self.samples = []

    def jotDown(self, state, action, reward, nState, tot):
        self.samples.append([state, action, reward, nState, tot])
        if len(self.samples) > self.size:
            self.samples.pop(0)

    def prepareData(self):
        samples = random.sample(self.samples, self.batchSize)
        states, actions, rewards, next_states, dones = zip(*samples)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def calculateReturn(self, discount):
        for reward in reversed(self.rewards):
            if len(self.returns) == 0:
                self.returns.append(reward)
            else:
                self.returns.append(reward + discount* self.returns[-1])

        self.returns.reverse()

    def calculateReturnFor(self, rewards, discount):
        returns = []
        for reward in reversed(rewards):
            if len(returns) == 0:
                returns.append(reward)
            else:
                returns.append(reward + discount* returns[-1])
        returns.reverse()
        return returns