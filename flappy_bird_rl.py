import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class FlappyBirdEnvironment:
    """
    Simplified Flappy Bird environment for RL training
    """
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.reset()
        
    def reset(self):
        """Reset the game state"""
        self.bird_x = 100
        self.bird_y = self.height // 2
        self.bird_speed_y = 0
        self.pipe_x = self.width
        self.pipe_y = random.randint(150, self.height - 150)
        self.pipe_gap = 200
        self.pipe_width = 60
        self.pipe_speed = 3
        self.score = 0
        self.game_over = False
        self.gravity = 0.5
        self.jump_speed = -8
        
        return self.get_state()
    
    def get_state(self):
        """Get current game state as feature vector"""
        # Normalize values to 0-1 range for better training
        bird_y_norm = self.bird_y / self.height
        bird_speed_norm = (self.bird_speed_y + 10) / 20  # Normalize speed
        pipe_x_norm = self.pipe_x / self.width
        pipe_y_norm = self.pipe_y / self.height
        
        # Distance to pipe opening (top and bottom)
        pipe_top = self.pipe_y - self.pipe_gap//2
        pipe_bottom = self.pipe_y + self.pipe_gap//2
        
        distance_to_pipe_top = (pipe_top - self.bird_y) / self.height
        distance_to_pipe_bottom = (pipe_bottom - self.bird_y) / self.height
        distance_to_pipe_x = (self.pipe_x - self.bird_x) / self.width
        
        return np.array([
            bird_y_norm,
            bird_speed_norm,
            distance_to_pipe_x,
            distance_to_pipe_top,
            distance_to_pipe_bottom,
            pipe_x_norm,
            pipe_y_norm
        ], dtype=np.float32)
    
    def step(self, action):
        """Execute one game step"""
        # Action: 0 = do nothing, 1 = jump
        if action == 1:
            self.bird_speed_y = self.jump_speed
        
        # Apply physics
        self.bird_speed_y += self.gravity
        self.bird_y += self.bird_speed_y
        
        # Move pipe
        self.pipe_x -= self.pipe_speed
        
        # Check if pipe passed
        reward = 0
        if self.pipe_x < self.bird_x and self.pipe_x + self.pipe_speed >= self.bird_x:
            reward = 100  # Reward for passing pipe
            self.score += 1
        
        # Reset pipe if it's off screen
        if self.pipe_x < -self.pipe_width:
            self.pipe_x = self.width
            self.pipe_y = random.randint(150, self.height - 150)
        
        # Check collisions
        bird_rect = (self.bird_x, self.bird_y, 30, 30)
        pipe_top_rect = (self.pipe_x, 0, self.pipe_width, self.pipe_y - self.pipe_gap//2)
        pipe_bottom_rect = (self.pipe_x, self.pipe_y + self.pipe_gap//2, self.pipe_width, self.height)
        
        # Collision detection
        if (self.bird_y <= 0 or self.bird_y >= self.height - 30 or
            self._rect_collision(bird_rect, pipe_top_rect) or
            self._rect_collision(bird_rect, pipe_bottom_rect)):
            self.game_over = True
            reward = -100  # Penalty for collision
        
        # Small reward for staying alive
        if not self.game_over:
            reward += 1
        
        return self.get_state(), reward, self.game_over, {'score': self.score}
    
    def _rect_collision(self, rect1, rect2):
        """Check if two rectangles collide"""
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        return (x1 < x2 + w2 and x1 + w1 > x2 and
                y1 < y2 + h2 and y1 + h1 > y2)


class DQN(nn.Module):
    """
    Deep Q-Network for Flappy Bird
    """
    def __init__(self, input_size=7, hidden_size=256, output_size=2):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class FlappyBirdAgent:
    """
    DQN Agent for Flappy Bird
    """
    def __init__(self, state_size=7, action_size=2, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = lr
        self.batch_size = 32
        self.target_update_freq = 1000
        self.steps = 0
        
        # Neural networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQN(state_size, 256, action_size).to(self.device)
        self.target_network = DQN(state_size, 256, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Copy weights to target network
        self.update_target_network()
        
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self):
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.99 * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.update_target_network()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filename):
        """Save the trained model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, filename)
    
    def load(self, filename):
        """Load a trained model"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']


class AIPersonality:
    """
    Different AI personalities with varying behaviors
    """
    def __init__(self, name, risk_factor=1.0, aggression=1.0, patience=1.0):
        self.name = name
        self.risk_factor = risk_factor  # How risky the AI plays
        self.aggression = aggression    # How often it takes action
        self.patience = patience        # How long it waits before acting
        
    def modify_action(self, base_action, state, agent):
        """Modify the base action based on personality"""
        # Conservative AI - less likely to jump
        if self.name == "Conservative":
            if base_action == 1 and random.random() > 0.7:
                return 0
        
        # Aggressive AI - more likely to jump
        elif self.name == "Aggressive":
            if base_action == 0 and random.random() > 0.8:
                return 1
        
        # Random AI - adds randomness
        elif self.name == "Chaotic":
            if random.random() < 0.1:
                return 1 - base_action
        
        return base_action


def train_agent(episodes=2000, save_path="flappy_bird_agent.pth"):
    """
    Train the RL agent
    """
    env = FlappyBirdEnvironment()
    agent = FlappyBirdAgent()
    
    scores = []
    avg_scores = []
    
    print("Starting training...")
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while not env.game_over and steps < 1000:  # Max steps per episode
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        agent.replay()
        scores.append(env.score)
        
        # Calculate running average
        if len(scores) >= 100:
            avg_score = np.mean(scores[-100:])
            avg_scores.append(avg_score)
        
        # Print progress
        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
            print(f"Episode {episode}, Average Score: {avg_score:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}, Steps: {steps}")
    
    # Save the trained agent
    agent.save(save_path)
    print(f"Training completed! Agent saved to {save_path}")
    
    # Plot training progress
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.title('Scores per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    
    plt.subplot(1, 2, 2)
    if avg_scores:
        plt.plot(avg_scores)
        plt.title('Average Score (100 episodes)')
        plt.xlabel('Episode (x100)')
        plt.ylabel('Average Score')
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.show()
    
    return agent


def create_ai_personalities():
    """
    Create different AI personalities for tournaments
    """
    personalities = [
        AIPersonality("Conservative", risk_factor=0.5, aggression=0.7, patience=1.5),
        AIPersonality("Aggressive", risk_factor=1.5, aggression=1.3, patience=0.7),
        AIPersonality("Balanced", risk_factor=1.0, aggression=1.0, patience=1.0),
        AIPersonality("Chaotic", risk_factor=1.2, aggression=0.9, patience=0.8),
        AIPersonality("Cautious", risk_factor=0.3, aggression=0.6, patience=2.0)
    ]
    return personalities


def run_tournament(agent_path="flappy_bird_agent.pth", rounds=10):
    """
    Run a tournament between different AI personalities
    """
    personalities = create_ai_personalities()
    env = FlappyBirdEnvironment()
    
    # Load trained agent
    base_agent = FlappyBirdAgent()
    base_agent.load(agent_path)
    
    results = {}
    
    print("Starting AI Tournament!")
    print("=" * 50)
    
    for personality in personalities:
        scores = []
        
        for round_num in range(rounds):
            state = env.reset()
            steps = 0
            
            while not env.game_over and steps < 1000:
                base_action = base_agent.act(state, training=False)
                action = personality.modify_action(base_action, state, base_agent)
                
                state, reward, done, info = env.step(action)
                steps += 1
                
                if done:
                    break
            
            scores.append(env.score)
        
        avg_score = np.mean(scores)
        max_score = max(scores)
        results[personality.name] = {
            'average': avg_score,
            'maximum': max_score,
            'scores': scores
        }
        
        print(f"{personality.name:12} - Avg: {avg_score:6.2f}, Max: {max_score:3d}")
    
    print("=" * 50)
    
    # Determine winner
    winner = max(results.keys(), key=lambda x: results[x]['average'])
    print(f"Tournament Winner: {winner} with average score of {results[winner]['average']:.2f}")
    
    return results


# Example usage and testing
if __name__ == "__main__":
    # Uncomment to train a new agent
    # trained_agent = train_agent(episodes=1000)
    trained_agent = train_agent(episodes=2000, save_path="flappy_bird_agent.pth")
    # Run tournament (assumes you have a trained agent)
    try:
        tournament_results = run_tournament(rounds=5)
    except FileNotFoundError:
        print("No trained agent found. Training new agent...")
        trained_agent = train_agent(episodes=500)
        tournament_results = run_tournament(rounds=5)