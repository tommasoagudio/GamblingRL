from sb3_contrib import QRDQN
import numpy as np

# Function to load the model
def load_model(model_path):
    return QRDQN.load("/Users/tommasoagudio/Documents/GitHub/GamblingRL/Risultati Alessio/QR-DQN1M.zip")

# Function to create the game state array
def create_game_state(player_hand_sum, dealer_visible_card, usable_ace):
    player_obs = [player_hand_sum] + [0] * 10  # Player's hand sum and padding
    dealer_obs = [dealer_visible_card] + [0] * 10  # Dealer's visible card and padding
    usable_ace_obs = [1] if usable_ace else [0]
    return np.array(player_obs + dealer_obs + usable_ace_obs)

# Function to predict the best move
def predict_move(model, game_state):
    action, _ = model.predict(game_state)
    return 'hit' if action == 0 else 'stick'

if __name__ == "__main__":
    model_path = 'your_model_path'  # Replace with your actual model path
    model = load_model(model_path)

    # Input prompts for user
    player_hand_sum = int(input("Enter player's hand sum: "))
    dealer_visible_card = int(input("Enter dealer's visible card: "))
    usable_ace_input = input("Does the player have a usable ace? (yes/no): ")
    usable_ace = usable_ace_input.lower() == 'yes'

    # Create the game state and predict the best move
    game_state = create_game_state(player_hand_sum, dealer_visible_card, usable_ace)
    best_move = predict_move(model, game_state)
    print(f"The best move is: {best_move}")
