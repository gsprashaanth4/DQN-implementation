## Deep Q-Network (DQN) Implementation from Scratch

This project is an educational implementation of Deep Q-networks - built from first principles, for discrete-action reinforcement learning. The program files include implementations of:

- experience replay
- epsilon-greedy action selection
- state normalization
- target Q-network
- Bellman target calculation
- mini-batch Q-value updates
- custom back propagation and optimization
- periodic target network synchronization
- model saving and greedy policy evaluation

### Demo

The implementation is demonstrated on discrete-action environments from Gymnasium, including CartPole-v1 and LunarLander-v3. The DQN implementation is built using my custom neural-network framework and is primarily intended as a learning implementation of DQN.

![DQN - LunarLander-v3 discrete](https://github.com/gsprashaanth4/DQN-implementation/blob/main/media/DQ_LL_%40.gif)

![DQN - CartPole-v1 discrete - 1st](https://github.com/gsprashaanth4/DQN-implementation/blob/main/media/DQ_CP_!.gif)

![DQN - CartPole-v1 discrete - 2nd](https://github.com/gsprashaanth4/DQN-implementation/blob/main/media/DQ_CP_%40.gif)