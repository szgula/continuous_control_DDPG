

from unityagents import UnityEnvironment
import numpy as np
import time
from ddpg_agent import myDDPG, myMADDPG
from collections import deque
import torch
import supporting_functions

REWARD_MULTIPLICATION = 1
EXPECTED_SCORE = 1 * REWARD_MULTIPLICATION
VISUALIZE = False
ENABLE_MADDPG = True


def ddpg(env, agent, brain_name, n_episodes=10000, max_t=300, print_every=10):
    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1, n_episodes+1):
        tick = time.time()
        env_info = env.reset(train_mode=not(VISUALIZE))[brain_name]
        states = env_info.vector_observations
        agent.reset()
        score = np.zeros((len(env_info.agents),))
        while True:
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            score += np.array(rewards)
            if any(dones):
                break 
                
        
        scores_deque.append(np.mean(score))
        scores.append(np.mean(score))
        episode_time = time.time() - tick
        print('\rEpisode {}\tScore: {:.2f}\ttime: {:.2f}'.format(i_episode, scores[-1], episode_time))
        
        if i_episode % print_every == 0:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            if len(scores_deque) == 100 and np.mean(scores_deque) >= EXPECTED_SCORE:
                print('Done! Yay!')
                break
            
    return scores, i_episode

def maddpg(env, agents, brain_name, n_episodes=10000, max_t=300, print_every=10):
    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1, n_episodes + 1):
        tick = time.time()
        env_info = env.reset(train_mode=not (VISUALIZE))[brain_name]
        states = env_info.vector_observations
        [agent.reset() for agent in agents]
        score = np.zeros((len(env_info.agents),))
        while True:
            actions = [agent.act(states[i]) for i, agent in enumerate(agents)]
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = [i * REWARD_MULTIPLICATION for i in env_info.rewards]
            dones = env_info.local_done
            [agent.step(states, actions, rewards, next_states, dones) for agent in agents]
            states = next_states
            score += np.array(rewards)
            if any(dones):
                break

        scores_deque.append(np.mean(score))
        scores.append(np.mean(score))
        episode_time = time.time() - tick
        print('\rEpisode {}\tScore: {:.2f}\ttime: {:.2f}'.format(i_episode, scores[-1], episode_time))

        if i_episode % print_every == 0:
            for agent_idx in range(agents[0].number_of_agents):
                torch.save(agents[agent_idx].actor_local.state_dict(), 'mmddpg_checkpoint_actor_{}.pth'.format(agent_idx))
                torch.save(agents[agent_idx].critic_local.state_dict(), 'mmddpg_checkpoint_critic_{}.pth'.format(agent_idx))
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            if len(scores_deque) == 100 and np.mean(scores_deque) >= EXPECTED_SCORE:
                print('Done! Yay!')
                break

    return scores, i_episode


def run_trained_agent(env, agent, brain_name):
    env_info = env.reset(train_mode=False)[brain_name]

    states = env_info.vector_observations
    agent.reset()
    while True:
        actions = agent.act(states)
        env_info = env.step(actions)[brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        states = next_states
        if any(dones):
            break 


def main():
    #env = UnityEnvironment(file_name='Reacher.app')
    env = UnityEnvironment(file_name="Tennis.app")

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=not(VISUALIZE))[brain_name]
    supporting_functions.print_env_info(env_info, brain)
    '''number of agents'''
    num_agents = len(env_info.agents)
    '''agent's actions size'''
    action_size = brain.vector_action_space_size
    states = env_info.vector_observations
    '''size of state observed by each agent'''
    obs_size = states.shape[1]
    '''total observed size - for all agents'''
    state_size = obs_size * states.shape[0]

    if not(ENABLE_MADDPG):
        agent = myDDPG(obs_size, action_size, 1, 0, enable_priority=False)
        scores, time_to_learn = ddpg(env, agent, brain_name)

    else:
        agents = [myMADDPG(obs_size, action_size, num_agents, 0, index_of_agent_in_maddpg=agent_idx) for agent_idx in range(num_agents)]
        [agent.set_other_agents(agents) for agent in agents]
        scores, time_to_learn = maddpg(env, agents, brain_name)

    title = 'score_{:.2f}_after_{}'.format(scores[-1], time_to_learn)
    supporting_functions.plot_average_score(scores, title)

    if not(ENABLE_MADDPG):
        run_trained_agent(env, agent, brain_name)

    supporting_functions.save_algorithm_parameters(agent, title, time_to_learn)


if __name__ == "__main__":
    main()



