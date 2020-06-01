import kuimaze
import numpy as np
import random


def learn_policy(env: kuimaze.HardMaze):
    """
    A function which handles basic reinforcement learning. (Q-Learning)
    Uses Epsilon-greedy strategy for selecting actions.
    :param env: A maze of type kuimaze.HardMaze.
    :return: policy: A dictionary, indexed by (x,y) tuples, with actions as values.
    """
    # The utility functions are defined.
    # max over a, selects the best value of the state
    # from all q-values.
    def max_a(q_values, s):
        return np.max(q_values[s[0]][s[1]])

    # Selects the best action.
    def argmax_action(q_values, s):
        return np.argmax(q_values[s[0]][s[1]])

    random.seed()

    # Using epsilon-greedy. Epsilon decreases with each episode.
    # Alpha is the learning rate.
    epsilon = 0.99
    alpha = 0.1
    discount = 0.9

    # The dimensions of the maze, and the number of actions possible is determined.
    x_dimension = env.observation_space.spaces[0].n
    y_dimension = env.observation_space.spaces[1].n
    num_actions = env.action_space.n

    # With these values, Q-values are initialized as 0.
    q_table = np.zeros([x_dimension, y_dimension, num_actions], dtype=float)

    max_episodes = 10000
    current_episode = 0

    # An agent can walk at most, all the tiles. If for some reason it
    # ends up in a loop, we should be able to terminate the episode.
    max_walk_length = x_dimension * y_dimension

    while current_episode < max_episodes:
        # Epsilon decreases with each new episode.
        epsilon *= 0.999
        current_episode += 1

        # The new episode is started, and relevant values are initialized.
        observation = env.reset()
        state = observation[0:2]
        done = False
        current_walk_length = 0

        while not done and current_walk_length < max_walk_length:
            current_walk_length += 1

            # Epsilon-greed. We get a uniform random ( [0, 1) ) variable,
            # and if it is lower than epsilon, we act randomly.
            is_random = random.random() < epsilon

            if is_random:
                action = env.action_space.sample()
            else:
                action = argmax_action(q_table, state)

            new_observation, reward, done, _ = env.step(action)
            new_state = new_observation[0:2]

            # Trial = R_t+1 + disc * max_a Q(S_t+1, a)
            trial = reward + discount * max_a(q_table, new_state)

            # Q(S_t, A_t) <- (1-alpha) Q(S_t, A_t) + (alpha) (Trial)
            previous_value = q_table[state[0]][state[1]][action]

            # Q-Update.
            q_table[state[0]][state[1]][action] = (1-alpha) * previous_value + alpha * trial
            state = new_state

    # Extract the policy from the q-table.
    policy = {}
    for i in range(x_dimension):
        for j in range(y_dimension):
            policy[(i, j)] = argmax_action(q_table, [i, j])

    return policy

