#!/usr/bin/env python3
import math
import gym
import random
import sys
import os
from bisect import bisect_left, bisect_right

class QLearner:
    filename = sys.argv[1]
    
    def __init__(self, buckets=6, alpha=0.01, gamma=0.01, exploration_rate = 0.1 ,exploration_min = 0.0, exploration_decay = 0.6, exploration_degrade = True, sarsa = True):
        self.rewards_all = []
        self.knowledge = dict()
                
        self.buckets = buckets
        self.alpha = alpha
        self.gamma = gamma
        self.exploration_rate = exploration_rate
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay
        self.exploration_degrade = exploration_degrade
        
        self.sarsa = sarsa
        
        self.environment = gym.make('CartPole-v1')
        self.attempt_no = 1
        self.upper_bounds = [
            self.environment.observation_space.high[0],
            0.5,
            self.environment.observation_space.high[2],
            math.radians(50)
        ]
        self.lower_bounds = [
            self.environment.observation_space.low[0],
            -0.5,
            self.environment.observation_space.low[2],
            -math.radians(50)
        ]

    def learn(self, max_attempts):
        for _ in range(max_attempts):
            reward_sum = self.attempt()
            # if reward_sum > 100:
            #     print(_, "=" * int(reward_sum / 100) + ">", reward_sum)
            # else:
            #     print(_, reward_sum)
            self.rewards_all.append(reward_sum)
            print(_, reward_sum)

    def attempt(self):
        observation = self.discretise(self.environment.reset())
        done = False
        reward_sum = 0.0
        while not done:
            self.knowledge.setdefault((observation, 0), 0.0)
            self.knowledge.setdefault((observation, 1), 0.0)
            # self.environment.render()
            action = self.pick_action(observation)
            new_observation, reward, done, info = self.environment.step(action)
            new_observation = self.discretise(new_observation)
            self.knowledge.setdefault((new_observation, 0), 0.0)
            self.knowledge.setdefault((new_observation, 1), 0.0)
            reward = -1 if done and reward_sum < 499 else 0
            if self.sarsa:
                self.update_knowledge_sarsa(action, observation, new_observation, reward)
            else:
                self.update_knowledge(action, observation, new_observation, reward)
            
            reward = 1
            observation = new_observation
            reward_sum += reward
        self.attempt_no += 1
        if self.exploration_degrade:
            self.exploration_rate *= self.exploration_decay
            self.exploration_rate = max(self.exploration_min, self.exploration_rate)
        return reward_sum
    
    def discretise(self, observation):
        bucket_count = 6;
        discrete = []
        for i in range(4):
            discrete.append(math.ceil((observation[i] - self.lower_bounds[i]) / (self.upper_bounds[i] - self.lower_bounds[i]) * bucket_count))
        return tuple(discrete)

    def pick_action(self, observation):
        if random.random() > 1 - self.exploration_rate:
            return self.environment.action_space.sample()
        else:
            return 0 if self.knowledge[(observation, 0)] > self.knowledge[(observation, 1)] else 1

    def update_knowledge(self, action, observation, new_observation, reward):
        self.knowledge[(observation,action)] = (1.0 - self.alpha) * self.knowledge[(observation, action)] + self.alpha * (reward + self.gamma * max(self.knowledge[(new_observation, 0)], self.knowledge[(new_observation, 1)]))

    def update_knowledge_sarsa(self, action, observation, new_observation, reward):
        self.knowledge[(observation,action)] = self.knowledge[(observation,action)] + self.alpha * (reward + self.gamma * self.knowledge[(new_observation, self.pick_action(observation))] - self.knowledge[(observation,action)] )

    def save(self, file):
        with open(file, "w") as File:
            File.write('\n'.join([str(num) for num in self.rewards_all]))
            
def main():
    try:
        learner = QLearner()
        learner.learn(10000)  
        learner.save(QLearner.filename)
        # print(learner.knowledge)
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            learner.save(QLearner.filename)
            # print(learner.knowledge)
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    


if __name__ == '__main__':
    main()
