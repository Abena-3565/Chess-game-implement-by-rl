import numpy as np
import random

# enviornment setup
BOARD_SIZE = 8
EMPTY = 0
PLAYER_PIECE = 1
OPPONENT_PIECE = -1
WIN_REWARD = 10
CAPTURE_REWARD = 1
MOVE_PENALTY = -0.1

class DamaEnv:
    def __init__(self):
        self.reset()

    def reset(self):
       
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.board[0:2, :] = OPPONENT_PIECE  
        self.board[6:8, :] = PLAYER_PIECE    
        self.done = False
        self.winner = None
        return self.get_state()
    
    def get_state(self):
    
        return self.board.flatten()
    
    def get_valid_actions(self, player):
        actions = []
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if self.board[row, col] == player:
                    moves = self.get_moves(row, col, player)
                    actions.extend(moves)
        return actions
    
    def get_moves(self, row, col, player):
        moves = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < BOARD_SIZE and 0 <= new_col < BOARD_SIZE:
                if self.board[new_row, new_col] == EMPTY:
                    moves.append(((row, col), (new_row, new_col)))  # normal move
                elif self.board[new_row, new_col] == -player:
                    # Check capture move
                    jump_row, jump_col = new_row + dr, new_col + dc
                    if 0 <= jump_row < BOARD_SIZE and 0 <= jump_col < BOARD_SIZE:
                        if self.board[jump_row, jump_col] == EMPTY:
                            moves.append(((row, col), (jump_row, jump_col)))  # capture
        return moves

    def step(self, action, player):
        (start_pos, end_pos) = action
        row, col = start_pos
        new_row, new_col = end_pos
        
        reward = MOVE_PENALTY
        if abs(new_row - row) == 2:
            # Capture move
            captured_row, captured_col = (row + new_row) // 2, (col + new_col) // 2
            self.board[captured_row, captured_col] = EMPTY
            reward = CAPTURE_REWARD
        
        # Move piece
        self.board[row, col] = EMPTY
        self.board[new_row, new_col] = player
        
        # Check for win condition
        if not np.any(self.board == OPPONENT_PIECE):
            reward = WIN_REWARD
            self.done = True
            self.winner = player
        
        return self.get_state(), reward, self.done

# Q-learning agent with state encoding and action encoding
class QLearningAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.99):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
    
    def encode_state(self, state):
        # Convert the numpy array state into a unique hashable format
        return hash(state.tobytes())

    def encode_action(self, action):
        # Generate a unique encoding for an action
        (start, end) = action
        return start[0] * 1000 + start[1] * 100 + end[0] * 10 + end[1]

    def choose_action(self, state, valid_actions):
        encoded_state = self.encode_state(state)
    
        if random.uniform(0, 1) < self.epsilon:
            print("Exploring")
            return random.choice(valid_actions)  # Exploration
        else:
            print("Exploiting")
            q_values = [self.q_table.get((encoded_state, self.encode_action(action)), 0) for action in valid_actions]
            max_q = max(q_values)
            return valid_actions[q_values.index(max_q)]


    def update_q_value(self, state, action, reward, next_state):
        encoded_state = self.encode_state(state)
        encoded_next_state = self.encode_state(next_state)
        action_idx = self.encode_action(action)
        
        # Get the maximum Q-value for the next state
        next_q_values = [
            self.q_table.get((encoded_next_state, self.encode_action(a)), 0)
            for a in env.get_valid_actions(PLAYER_PIECE)
        ]
        best_next_q = max(next_q_values) if next_q_values else 0
        
        # Update Q-value using the Q-learning formula
        current_q = self.q_table.get((encoded_state, action_idx), 0)
        td_target = reward + self.gamma * best_next_q
        self.q_table[(encoded_state, action_idx)] = current_q + self.alpha * (td_target - current_q)

    def decay_epsilon(self):
        # Gradually decay epsilon to reduce exploration over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

# Training loop
env = DamaEnv()
agent = QLearningAgent(state_size=BOARD_SIZE * BOARD_SIZE, action_size=BOARD_SIZE * BOARD_SIZE * 10)

MAX_STEPS = 20  # Add a maximum step limit to prevent infinite episodes
for episode in range(10):
    state = env.reset()
    total_reward = 0
    player = PLAYER_PIECE  # Start with the player
    step = 0
    while not env.done and step < MAX_STEPS:
        if player == PLAYER_PIECE:
            # Player's turn
            valid_actions = env.get_valid_actions(PLAYER_PIECE)
            if not valid_actions:
                break
            action = agent.choose_action(state, valid_actions)
            next_state, reward, done = env.step(action, PLAYER_PIECE)
            agent.update_q_value(state, action, reward, next_state)
            state = next_state
            total_reward += reward

            # Print only after PLAYER_PIECE action
            print(f"Action taken: {action}, Reward: {reward}")
        
        else:
            # Opponent's turn (simple random strategy)
            valid_actions = env.get_valid_actions(OPPONENT_PIECE)
            if valid_actions:
                opponent_action = random.choice(valid_actions)
                env.step(opponent_action, OPPONENT_PIECE)
        
        # Switch turn to the other player
        player = PLAYER_PIECE if player == OPPONENT_PIECE else OPPONENT_PIECE
        step += 1
    
    # Only print once per episode
    agent.decay_epsilon()
    print(f"Episode {episode + 1}: Total Reward = {total_reward}, Epsilon = {agent.epsilon:.4f}")

print("Training completed.")
