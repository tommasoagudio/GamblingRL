import random
import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import DQN
from gym.wrappers import monitoring

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


#defining the model
def evaluate_agent(model, env, num_games):
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
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    gamma = trial.suggest_float('gamma', 0.9, 0.9999)
    buffer_size = trial.suggest_int('buffer_size', 10000, 1000000)
    exploration_fraction = trial.suggest_float('exploration_fraction', 0.1, 0.5)
    
    env = DummyVecEnv([lambda: SimpleBlackjackEnv()])
    
    model = DQN(
        "MlpPolicy", 
        env, 
        learning_rate=learning_rate, 
        gamma=gamma, 
        buffer_size=buffer_size, 
        exploration_fraction=exploration_fraction,
        verbose=0
    )
    
    model.learn(total_timesteps=100000)
    
    mean_reward= evaluate_agent(model, env, num_games=1000)
    
    return -mean_reward

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)
    
    best_params = study.best_params
    best_reward = -study.best_value

    
    env = DummyVecEnv([lambda: SimpleBlackjackEnv()])
    best_model = DQN("MlpPolicy", env, **best_params, verbose=1,tensorboard_log="./dqn_blackjack_tensorboard/")
    best_model.learn(total_timesteps=200000)
    
    # Save the best model
    best_model.save("dqn_blackjack")