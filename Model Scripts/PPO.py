import random
import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

#defining the env
def cmp(a, b):
    if a > b:
        return 1
    elif a < b:
        return -1
    else:
        return 0

class SimpleBlackjackEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(SimpleBlackjackEnv, self).__init__()
        self.deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4  # a full deck
        random.shuffle(self.deck)  # shuffle the deck
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=11, shape=(23,), dtype=int)
        
    def draw_card(self):
        return self.deck.pop()
        
    def draw_hand(self):
        return [self.draw_card(), self.draw_card()]

    def usable_ace(self, hand):
        return 1 in hand and sum(hand) + 10 <= 21

    def sum_hand(self, hand):
        if self.usable_ace(hand):
            return sum(hand) + 10
        return sum(hand)

    def is_bust(self, hand):
        return self.sum_hand(hand) > 21

    def score(self, hand):
        return 0 if self.is_bust(hand) else self.sum_hand(hand)

    def reset(self):
        if len(self.deck) < 15:
            self.deck = [1, 2, 3, 4, 5, 6, 7, 8, 9,
                         10, 10, 10, 10] * 4
            random.shuffle(self.deck)
        self.dealer = self.draw_hand()
        self.player = self.draw_hand()
        return self._get_observation()

    def step(self, action):
        assert self.action_space.contains(action)
        if action == 1:  # hit
            self.player.append(self.draw_card())
            if self.is_bust(self.player):
                done = True
                reward = -1.0
            else:
                done = False
                reward = 0.0
        else:  # stick
            done = True
            while self.sum_hand(self.dealer) < 17:
                self.dealer.append(self.draw_card())
            reward = cmp(self.score(self.player), self.score(self.dealer))
        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        player_obs = self.player + [0] * (11 - len(self.player))
        dealer_obs = self.dealer + [0] * (11 - len(self.dealer))
        usable_ace_obs = [1] if self.usable_ace(self.player) else [0]
        return np.array(player_obs + dealer_obs + usable_ace_obs)

    def render(self, mode='human'):
        if mode != 'human':
            raise NotImplementedError()
        return f"Player hand: {self.player}, Dealer hand: {self.dealer}"

    def close(self):
        pass

# Testing the environment to ensure it initializes and steps correctly
env = SimpleBlackjackEnv()
obs = env.reset()
print(env.render())
obs, reward, done, _ = env.step(1)
print(env.render())
obs, reward, done, _ = env.step(0)
print(env.render())

#creating the model with optuna
def evaluate_agent(model, env, num_games=1000): #
    wins = 0                                    
    for _ in range(num_games):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
        if reward == 1.0:
            wins += 1
    win_rate = wins / num_games
    return win_rate

def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.9999)
    n_steps = trial.suggest_int("n_steps", 16, 2048, log=True)
    ent_coef = trial.suggest_float("ent_coef", 1e-8, 1e-1, log=True)
    
    env = DummyVecEnv([lambda: SimpleBlackjackEnv()])
    model = PPO("MlpPolicy", env, learning_rate=learning_rate, gamma=gamma, n_steps=n_steps, ent_coef=ent_coef, verbose=0)
    
    model.learn(total_timesteps=100)
    
    win_rate = evaluate_agent(model, env)
    
    return -win_rate  

study = optuna.create_study()
study.optimize(objective, n_trials=3)  #3 trials due to computational limits

print(study.best_params) # <-- best hyperparameters found (not necessarily the same as best model)


#training the agent with the best hyperparameters found by optuna
best_params = study.best_params
env = DummyVecEnv([lambda: SimpleBlackjackEnv()])
model = PPO("MlpPolicy", env, **best_params, verbose=1, tensorboard_log="./ppo_blackjack_tensorboard/")
model.learn(total_timesteps=100)
model.save("ppo_blackjack")