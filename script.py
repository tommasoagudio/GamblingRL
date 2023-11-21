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
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor
import pandas as pd
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
        self.reward_range = (-np.inf, np.inf)
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
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

# Testing the environment to ensure it initializes and steps correctly
env = SimpleBlackjackEnv()
obs = env.reset()
print(env.render())
obs, reward, done, _ = env.step(1)
print(env.render())
obs, reward, done, _ = env.step(0)
print(env.render())
def evaluate_agent(model, env, num_games=1000):
    wins = 0
    win_rates = []
    num_games_list = []  # List to store the number of games after each logging interval

    for i in range(num_games):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            if done and reward == 1:
                wins += 1
        if (i + 1) % 100 == 0:  # Log win rate every 100 games
            win_rates.append(wins / (i + 1))
            num_games_list.append(i + 1)  # Append the number of games played so far

    # Create a DataFrame with both win rates and number of games
    win_rate_df = pd.DataFrame({'WinRate': win_rates, 'NumGames': num_games_list})
    win_rate_df.to_csv('PPO500k_win_rate_over_time.csv', index=False)
    
    return wins / num_games


# Create the environment
env = DummyVecEnv([lambda: SimpleBlackjackEnv()])

# Set hyperparameters
params = {
    'learning_rate': 2.5e-4,
    'n_steps': 256,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 1e-4
}

# Instantiate the model
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="./blackjack_tensorboard/", **params)

# Train the model
model.learn(total_timesteps=100)

# Evaluate the model
win_rate = evaluate_agent(model, env)

def simulate_blackjack_games(env, model, num_games=10000):
    action_frequencies = {}
    rewards = []
    results = []

    for game in range(num_games):
        obs = env.reset()
        done = False
        total_reward = 0
        player_actions = []
        player_hand_sums = []

        while not done:
            action, _ = model.predict(obs)
            player_actions.append('Hit' if action == 1 else 'Stick')

            # Define state key
            player_hand = obs[:11][obs[:11] != 0]
            dealer_visible_card = env.dealer[0]
            state_key = (tuple(player_hand), dealer_visible_card)

            # Record action frequencies
            if state_key not in action_frequencies:
                action_frequencies[state_key] = {'Hit': 0, 'Stick': 0}
            action_frequencies[state_key]['Hit' if action == 1 else 'Stick'] += 1

            obs, reward, done, _ = env.step(action)
            total_reward += reward
            player_hand_sums.append(env.sum_hand(player_hand))

        rewards.append(total_reward)

        player_final_hand = obs[:11][obs[:11] != 0]
        dealer_final_hand = obs[11:22][obs[11:22] != 0]

        game_results = {
            'Game': game + 1,
            'PlayerFinalHandSum': env.sum_hand(player_final_hand),
            'DealerFinalHandSum': env.sum_hand(dealer_final_hand),
            'PlayerNumCards': len(player_final_hand),
            'DealerNumCards': len(dealer_final_hand),
            'DealerVisibleCard': dealer_visible_card,
            'PlayerActions': ' '.join(player_actions),
            'PlayerHandProgression': ' '.join(map(str, player_hand_sums)),
            'Outcome': 'Win' if reward > 0 else 'Loss' if reward < 0 else 'Draw'
        }
        results.append(game_results)

    # Export action frequencies and rewards
    action_freq_data = []
    for state, actions in action_frequencies.items():
        player_hand, dealer_card = state
        action_freq_data.append({'PlayerHand': ' '.join(map(str, player_hand)), 
                                 'DealerVisibleCard': dealer_card,
                                 'Hit': actions['Hit'], 
                                 'Stick': actions['Stick']})
    
    action_freq_df = pd.DataFrame(action_freq_data)
    action_freq_df.to_csv('PPO500k_action_frequencies.csv', index=False)
    
    rewards_df = pd.DataFrame(rewards, columns=['Reward'])
    rewards_df.to_csv('PPO500k_rewards_distribution.csv', index=False)

    results_df = pd.DataFrame(results)
    results_df.to_csv('PPO500k_blackjack_results.csv', index=False)

    win_rate = results_df[results_df['Outcome'] == 'Win'].shape[0] / num_games
    print(f"\nAgent won {results_df[results_df['Outcome'] == 'Win'].shape[0]} out of {num_games} games. Win rate: {win_rate * 100:.2f}%")
    return win_rate

model.save("PPO500k")
env = SimpleBlackjackEnv()
simulate_blackjack_games(env, model)
print(f"Win rate: {win_rate:.2f}")