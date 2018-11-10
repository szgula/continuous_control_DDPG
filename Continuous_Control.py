

from unityagents import UnityEnvironment
import numpy as np
import time
from ddpg_agent import Agent
from collections import deque
import torch
import supporting_functions

ENABLE_PRIORITY =   False
BUFFER_SIZE     =   int(1e5)
BATCH_SIZE      =   128
GAMMA           =   0.99
TAU             =   1e-3
LR_ACTOR        =   1e-5
LR_CRITIC       =   1e-4
WEIGHT_DECAY    =   0


def ddpg(env, agent, brain_name, n_episodes=1000, max_t=300, print_every=1):
    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1, n_episodes+1):
        tick = time.time()
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        agent.reset()
        score = np.zeros((20,))
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
        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
        
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            if len(scores_deque) == 100 and np.mean(scores_deque) >= 30:
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
    env = UnityEnvironment(file_name='Reacher.app')

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    supporting_functions.print_env_info(env_info, brain)
    env_info.vector_observations.shape

    agent = Agent(33, 4, 0, enable_priority=False)

    scores, time_to_learn = ddpg(env, agent, brain_name)

    title = 'score_{:.2f}_after_{}'.format(scores[-1], time_to_learn)
    supporting_functions.plot_average_score(scores, title)

    run_trained_agent(env, agent, brain_name)

    supporting_functions.save_algorithm_parameters(agent, title, time_to_learn)


if __name__ == "__main__":
    main()



