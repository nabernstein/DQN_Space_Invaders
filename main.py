import gym
from gym.wrappers import Monitor
import os
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from time import time
from collections import deque
from math import ceil
from shutil import rmtree

from dqn_agent import DQN_Agent
from frame_stack import FrameStack
from replay_memory import ReplayMemory
from datetime import datetime

def reset_skip(env, frames, reset = False):
    transition = (None,)
    if reset:
        transition = env.reset()
    for i in range(frames):
        transition = env.step(0)
        if transition[2]:
            break
    return transition

def main():
    t_0 = time()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            tf.config.set_soft_device_placement(True)
        except RuntimeError as e:
            print(e)

    num_frames = 4
    env = gym.make("SpaceInvaders-v0")
    observation = env.reset()
    frame_stack = FrameStack(num_frames)
    frame_stack.add(observation)
    curr_state = frame_stack.get_state()
    input_shape = curr_state.shape

    learning_rate = 0.00025
    gamma = 0.99
    num_actions = env.action_space.n
    agent = DQN_Agent(input_shape, num_actions, learning_rate, gamma)

    mem_capacity = 150_000
    mem_minimum = 50_000
    minibatch_size = 32

    replay_memory = ReplayMemory(mem_capacity)

    train_for = 1_500_000
    epsilon = 0.1
    e_min = 0.1
    # e_decay = e_min ** (1 / (float(0.975*train_for)))
    e_decay = (epsilon - e_min) / (1e6)
    epsilon = 0.05
    save_every = 50_000

    pwd = datetime.now().strftime("./%y%m%d-%H%M%S-") + str(train_for)
    if not os.path.isdir(pwd):
        os.mkdir(pwd)
    model_dir = pwd + '/models/'
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    rec_dir = pwd + '/recordings/'
    if not os.path.isdir(rec_dir):
        os.mkdir(rec_dir)

    load_fname = "./si_agent_26h42m.h5"
    agent = DQN_Agent(input_shape, num_actions, learning_rate, gamma, load_fname)
    np.random.seed(int(time() + os.getpid()))
    # np.random.seed(123)

    observation = reset_skip(env, 40)[0]
    frame_stack.reset(observation)
    curr_state = frame_stack.get_state()
    lives = env.ale.lives()
    for frame in tqdm(range(1, mem_minimum+1),
                      desc="Populating Memory", ascii=True, unit='frame'):
        if np.random.random() > epsilon:
            action = np.argmax(agent.get_qs(curr_state))
        else:
            action = np.random.randint(0, num_actions)

        reward = 0
        done = False
        for i in range(3):
            observation, temp_reward, temp_done, _ = env.step(action)
            reward += temp_reward
            done = done | temp_done
            if done:
                break

        if env.ale.lives() < lives and (not done):
            lives = env.ale.lives()
            observation, reward, done, _ = reset_skip(env, 40)
            frame_stack.reset(observation)
        else:
            frame_stack.add(observation)

        next_state = frame_stack.get_state() if not done else np.zeros(input_shape)

        replay_memory.add(curr_state, action, reward, next_state, done)
        curr_state = next_state

        if done:
            observation = reset_skip(env, 40, reset=True)[0]
            frame_stack.reset(observation)
            curr_state = frame_stack.get_state()
            lives = env.ale.lives()

    episode_scores = deque(maxlen=10)
    episode_survived = deque(maxlen=10)
    episode_score = 0
    episode_steps = 0
    episode_count = 0
    steps = 0
    avg_desc = "E:%d - Best: %d - Avg(S:%d-F:%d) - (e%.4f) - "
    avg_desc_str = avg_desc % (0, 0, 0, 0, epsilon)
    episode_desc = "Cur(S:%d-F:%d-L:%d)"
    episode_desc_str = episode_desc % (0, 0, 0)
    pbar_desc = avg_desc_str + episode_desc_str

    logfile = pwd + '/log.txt'
    with open(logfile, "a") as myfile:
        myfile.write("Episode\tScore\tSurvived\n")

    if not os.path.isdir(rec_dir + 'tempdir'):
        os.mkdir(rec_dir + 'tempdir')
    env = gym.make("SpaceInvaders-v0")
    env = Monitor(env, rec_dir + 'tempdir', force=True)
    observation = reset_skip(env, 40, reset=True)[0]
    frame_stack.reset(observation)
    curr_state = frame_stack.get_state()
    lives = env.ale.lives()

    pbar = tqdm(total=train_for, desc=pbar_desc, ascii=True, unit='frame')
    best = 0
    while steps < train_for:

        if np.random.random() > epsilon:
            action = np.argmax(agent.get_qs(curr_state))
        else:
            action = np.random.randint(0, num_actions)

        reward = 0
        done = False
        for i in range(3):
            observation, temp_reward, temp_done, _ = env.step(action)
            reward += temp_reward
            done = done | temp_done
            if done:
                break

        episode_score += reward

        if env.ale.lives() < lives and (not done):
            lives = env.ale.lives()
            observation, reward, done, _ = reset_skip(env, 40)
            frame_stack.reset(observation)
        else:
            frame_stack.add(observation)

        next_state = frame_stack.get_state() if not done else np.zeros(input_shape)

        replay_memory.add(curr_state, action, reward, next_state, done)
        minibatch = replay_memory.sample(minibatch_size)
        agent.train(minibatch, steps)
        steps += 1
        episode_steps += 1
        pbar.update(1)
        episode_desc_str = episode_desc % (episode_score, episode_steps, lives)
        curr_state = next_state
        if done:
            episode_count += 1
            episode_scores.append(episode_score)
            episode_survived.append(episode_steps)
            if episode_score > best:
                best = episode_score

            with open(logfile, "a") as myfile:
                myfile.write("%d\t%d\t%d\n" % (episode_count, episode_score, episode_steps))
            avg_score = int(sum(episode_scores) / len(episode_scores))
            avg_step = int(sum(episode_survived) / len(episode_survived))
            avg_desc_str = avg_desc % (episode_count, best, avg_score, avg_step, epsilon)

            i = 1
            vid_dir = "%d_%d"
            while os.path.isdir(rec_dir + vid_dir % (episode_score, i)):
                i += 1

            del env
            if episode_score > 900:
                os.rename(rec_dir + 'tempdir', rec_dir + vid_dir % (episode_score, i))
            else:
                rmtree(rec_dir + 'tempdir')

            if not os.path.isdir(rec_dir + 'tempdir'):
                os.mkdir(rec_dir + 'tempdir')
            env = gym.make("SpaceInvaders-v0")
            env = Monitor(env, rec_dir + 'tempdir', force=True)
            observation = reset_skip(env, 40, reset=True)[0]
            frame_stack.reset(observation)
            curr_state = frame_stack.get_state()
            lives = env.ale.lives()
            episode_score = 0
            episode_steps = 0

        if epsilon > e_min:
            epsilon = max(e_min, epsilon-e_decay)
        if steps > 0 and (steps % save_every == 0):
            agent.save(model_dir + ("si_agent_%ds.h5" % steps))
        pbar.set_description(avg_desc_str + episode_desc_str)
    elapsed = time() - t_0
    hours, rem = divmod(elapsed, 3600)
    mins = int(ceil(rem / 60))

    train_time = "%dh%dm" % (hours, mins)

    model_fname = 'si_agent_' + train_time + '.h5'
    agent.save(model_dir + model_fname)


if __name__ == '__main__':
    main()