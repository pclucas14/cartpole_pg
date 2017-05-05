#!/usr/bin/env python

'''
code adapted from https://gist.github.com/shanest/535acf4c62ee2a71da498281c2dfc4f4
changed a few hyperparams and configured it for Lasagne.
'''

import gym
import theano
import theano.tensor as T
import lasagne
import lasagne.layers as  ll
import pdb
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

#hyperparameters 
learning_rate = 0.1
hidden_size = 50
num_actions = 2
input_size = 4
num_batches = 40
episodes_per_batch = 10 
downcast = True

srng = RandomStreams(seed=np.random.randint(2147462579))


class Agent():
    def __init__(self):
        model = []
        self.input_var = T.matrix('inputs')   # boards
        self.target_var = T.ivector('targets') # actions
        self.advantage = T.vector('advantage') # reward

        model.append(ll.InputLayer(shape=(None, input_size), input_var=self.input_var))

        model.append(ll.DenseLayer(model[-1], 
                     num_units=hidden_size))
                     
        model.append(ll.DenseLayer(model[-1], 
                     num_units=num_actions, 
                     nonlinearity=lasagne.nonlinearities.softmax))

        self.model = model
        self.params = ll.get_all_params(model[-1], trainable=True)

        probs = ll.get_output(model[-1])
        log_probs = T.log(probs)

        # sample action from probabilities
        sample = T.argmax(srng.multinomial(pvals=probs))
        self.sample = theano.function([self.input_var], 
                                    [probs,sample])
        
        # normalize advantage
        normalized_adv = (self.advantage - T.mean(self.advantage)) / (T.std(self.advantage) + 1e-10)

        ce_loss = lasagne.objectives.categorical_crossentropy(probs, self.target_var)
        pg_loss = (ce_loss * normalized_adv).mean()
        updates = lasagne.updates.rmsprop(pg_loss, self.params, learning_rate=learning_rate)

        self.train = theano.function([self.input_var, self.target_var, self.advantage],
                                    [pg_loss, normalized_adv], 
                                    updates=updates,
                                    allow_input_downcast=downcast)

    def choose_action(self, game_input):
        probs, action = self.sample(game_input.reshape((1,-1)))
        #pdb.set_trace()
        return action


def policy_rollout(env, agent):
    """Run one episode."""

    observation, reward, done = env.reset(), 0, False
    obs, acts, rews = [], [], []

    while not done:

        env.render()
        obs.append(observation)

        action = agent.choose_action(observation)
        observation, reward, done, _ = env.step(action)

        acts.append(action)
        rews.append(reward)

    return obs, acts, rews


def process_rewards(rews):
    """Rewards -> Advantages for one episode. """

    # total reward: length of episode
    return [len(rews)] * len(rews)


def main():

    env = gym.make('CartPole-v0')

    monitor_dir = '/tmp/cartpole_exp1'
    env = gym.wrappers.Monitor(env, monitor_dir, force=True)
    agent = Agent()

    for batch in xrange(num_batches):

        print '=====\nBATCH {}\n===='.format(batch)

        b_obs, b_acts, b_rews = [], [], []

        for _ in xrange(episodes_per_batch):

            obs, acts, rews = policy_rollout(env, agent)

            print 'Episode steps: {}'.format(len(obs))

            b_obs.extend(obs)
            b_acts.extend(acts)

            advantages = process_rewards(rews)
            b_rews.extend(advantages)

        # update policy
        # normalize rewards; don't divide by 0
        # b_rews = (b_rews - np.mean(b_rews)) / (np.std(b_rews) + 1e-10)

        loss, adv = agent.train(b_obs, b_acts, b_rews)
    env.close()


if __name__ == "__main__":
    main()
