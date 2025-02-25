import numpy as np
import matplotlib.pyplot as plt


class MazeEnvironment:
    """Optimized Maze Environment"""

    def __init__(self, maze, start, goal):
        self.maze = maze
        self.start = start
        self.goal = goal
        self.state = start
        self.actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    def reset(self):
        """Reset environment to initial state"""
        self.state = self.start
        return self.state

    def step(self, action):
        """Take action, return next state, reward, and done flag"""
        row, col = self.state
        d_row, d_col = self.actions[action]
        new_row, new_col = row + d_row, col + d_col

        # Validate move
        if (0 <= new_row < self.maze.shape[0] and
                0 <= new_col < self.maze.shape[1] and
                self.maze[new_row, new_col] == 1):
            self.state = (new_row, new_col)
            reward = -1
        else:
            reward = -10  # Hitting a wall

        # Check if goal is reached
        done = self.state == self.goal
        if done:
            reward = 50

        return self.state, reward, done

    def render(self):
        """Visualize maze with agent"""
        maze_copy = np.copy(self.maze)
        row, col = self.state
        maze_copy[row, col] = 2
        plt.imshow(maze_copy, cmap='gray')
        plt.show()


class QAgent:
    """Optimized Q-learning Agent"""

    def __init__(self, env, learning_rate=0.9, discount_factor=0.8, exploration_prob=0.2, decay_rate=0.95):
        self.env = env
        self.q_table = np.zeros((env.maze.shape[0], env.maze.shape[1], len(env.actions)))
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_prob
        self.decay_rate = decay_rate

    def choose_action(self, state):
        """Optimized epsilon-greedy action selection"""
        if np.random.rand() < self.epsilon:
            return np.random.choice(len(self.env.actions))
        else:
            row, col = state
            return np.argmax(self.q_table[row, col])

    def update_q_table(self, state, action, reward, next_state):
        """Vectorized Q-table update"""
        row, col = state
        next_row, next_col = next_state
        best_next_action = np.max(self.q_table[next_row, next_col])

        self.q_table[row, col, action] += self.lr * (
                    reward + self.gamma * best_next_action - self.q_table[row, col, action])

    def train(self, episodes=1):
        """Train agent for given number of episodes"""
        rewards_per_episode = []

        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            step_count = 0

            while not done and step_count < 200:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                self.update_q_table(state, action, reward, next_state)

                state = next_state
                total_reward += reward
                step_count += 1

            self.epsilon *= self.decay_rate
            rewards_per_episode.append(total_reward)

            print(f"Episode {episode}: Reward = {total_reward}, Steps = {step_count}")

        return rewards_per_episode

    def test(self):
        """Test trained agent"""
        state = self.env.reset()
        done = False
        steps = 0

        while not done and steps < 200:
            action = np.argmax(self.q_table[state])
            next_state, _, done = self.env.step(action)
            state = next_state
            steps += 1

            self.env.render()

        print(f"Reached goal in {steps} steps")

maze_grid = np.array([
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
    [0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
    [0, 1, 0, 1, 1, 1, 0, 1, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    [0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0]
])

start_pos = (0, 0)
goal_pos = (9, 9)

env = MazeEnvironment(maze_grid, start_pos, goal_pos)
agent = QAgent(env)

rewards = agent.train(episodes=2)
agent.test()
