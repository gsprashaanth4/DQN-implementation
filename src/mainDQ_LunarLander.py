from req.DQNNN import DQNNN
from req.Replay2 import Replay2
import numpy as np
import gymnasium as gym
import time
from copy import deepcopy
import sys
import os

def normalizeState(state):
    return np.array([
        state[0] / 2.5,
        state[1] / 2.5,
        state[2] / 10.0,
        state[3] / 10.0,
        state[4] / 6.2831855,
        state[5] / 10.0,
        state[6] / 1.0,
        state[7] / 1.0
    ])

env = gym.make("LunarLander-v3", render_mode=None)

if os.path.exists("models/lander.npz"):
    os.remove("models/lander.npz")
    print("deleted old NN")
    time.sleep(0.5)

QValueNetwork = DQNNN([8, 64, 64, 4])
targetNetwork = deepcopy(QValueNetwork)

batchSize = 128
replay = Replay2(batchSize)

gamma = 0.99

episodes = 700
copyAt = 700
print(copyAt)
time.sleep(0.5)
epsilon = 1.0
actionSpace = [0, 1, 2, 3]

trainStep = 0
MSEloss = 0
minMSEloss = sys.maxsize
maxReward = -100000000
sr = 0.0 

for episode in range(episodes):
    epsilon = max(0.05, epsilon * 0.995)
    observation, info = env.reset()
    observation = normalizeState(observation)
    terminated = False
    truncated = False
    rewards = []
    episodeReward = 0.0

    while not (terminated or truncated):

        qValues = QValueNetwork.interact(observation)

        # eplison-greedy sample collection
        if np.random.random() < epsilon:
            action = np.random.choice(actionSpace)
        else:
            action = np.argmax(qValues[0])

        prevObs = observation
        observation, reward, terminated, truncated, info = env.step(action)
        obsOrig = observation
        observation = normalizeState(observation)

        reward += (
            -float(abs(obsOrig[0])) * 4.0
            -float(abs(obsOrig[4])) * 2.0
            -float(abs(obsOrig[1])) * 0.8
            -float(abs(obsOrig[3])) * 0.3
        )
        episodeReward+=reward

        replay.jotDown(prevObs, action, reward, observation, terminated)

        if (len(replay.samples) < 1000):
            continue
        else:
            states, actions, rewards, next_states, dones = replay.prepareData()
            Qvals, _ = QValueNetwork.getQvals(states)
            nQvals, _ = targetNetwork.getQvals(next_states)
            targQvals = np.max(nQvals, axis=1)

            target = (rewards + gamma * (1-dones) * targQvals)

            # loss
            predQvals = Qvals[np.arange(len(actions)), actions]
            sqErrorPerSample = (predQvals - target) ** 2
            MSEloss = np.sum(sqErrorPerSample) / batchSize
            # print(episode, MSEloss, epsilon)

            dLdP = (2/batchSize) * (predQvals - target)
            dPdZ = np.zeros_like(Qvals)
            dPdZ[np.arange(batchSize), actions] = dLdP

            QValueNetwork.backward(dPdZ)
            QValueNetwork.optimize()

            trainStep+=1
            if trainStep % copyAt == 0:
                targetNetwork = deepcopy(QValueNetwork)
                print(" .", end="")
                # time.sleep(0.5)

    print("\n", f"episode: {episode:03}", f" loss: {MSEloss:12.6f}", f" epsilon: {epsilon:8.6f}", f" reward: {episodeReward:6.3f}", end="")

    if episodeReward > maxReward:
        maxReward = episodeReward
        QValueNetwork.save("models/lander.npz")
        print(" noted", end="")


env.close()

env = gym.make("LunarLander-v3", render_mode='human')

print
print("\n\nTesting Greedy")
QValueNetwork.load("models/lander.npz")
print("model loaded")
time.sleep(2)

for e in range(17):
    observation, info = env.reset()
    observation = normalizeState(observation)

    terminated = False
    truncated = False

    rewards = []
    episodeReward = 0

    while not (truncated or terminated):
        qValues = QValueNetwork.interact(observation)
        action = np.argmax(qValues[0])
        observation, reward, terminated, truncated, info = env.step(action)
        observation = normalizeState(observation)
        episodeReward+=reward

    print(e, episodeReward)

env.close()
