
from unityagents import UnityEnvironment
import matplotlib.pyplot as plt
import numpy as np

def print_env_info(env_info, brain):
    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])


def plot_average_score(scores, title):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

    fig.savefig('{}.png'.format(title))


def save_algorithm_parameters(agent, title, time_to_learn):
    with open('{}.txt'.format(title), "w") as f:
        text = 'Learning rate actor: {}'.format(agent.LR_ACTOR)
        f.write(text)

        text = 'Learning rate critic: {}'.format(agent.LR_CRITIC)
        f.write(text)

        text = 'Learning rate actor: {}'.format(agent.LR_ACTOR)
        f.write(text)