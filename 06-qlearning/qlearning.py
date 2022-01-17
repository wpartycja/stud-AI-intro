from email.charset import SHORTEST
import numpy as np
import matplotlib.pyplot as plt

WALL = '#'
FREE = '.'


class Qlearning_maze():
    def __init__(self, maze_path, goal, lr, discount_factor, episodes, epsilon=None):
        self.goal = goal
        self.epsilon = epsilon
        self.lr = lr
        self.discount_factor = discount_factor
        self.episodes = episodes
        self.maze = self.set_maze(maze_path)
        self.nrows, self.ncol = self.maze.shape
        self.actions = ['up', 'right', 'down', 'left']
        self.q_values = np.zeros((self.nrows, self.ncol, len(self.actions)))
        self.rewards_table = self.set_rewards()

        if self.goal[0] > self.nrows - 1 or self.goal[1] > self.ncol:
            raise IndexError("Size of the maze is too small for this coordinates")

    def set_maze(self, maze_path):
        """
        read teh maze from .txt file into numpy array
        """
        maze = np.loadtxt(maze_path, dtype=str, ndmin=2, comments="/")
        new_maze = []
        for i in range(maze.shape[0]):
            new_row = list(maze[i][0])
            new_maze.append(new_row)
        goal_x, goal_y = self.goal
        if new_maze[goal_x][goal_y] != FREE:
            raise ValueError("Goal can't be on a path which is WALL")
        new_maze[goal_x][goal_y] = 'F'
        return np.asarray(new_maze)
    
    def show_maze(self):
        print(self.maze)

    def set_rewards(self):
        """
        sets reward table
        """
        rewards = np.full((self.nrows, self.ncol), -100)
        x, y = self.goal
        rewards[x][y] = 100

        for y in range(self.ncol):
            for x in range(self.nrows):
                if self.maze[x][y] == FREE:
                    rewards[x][y] = -1

        return rewards

    def is_illegal(self, x, y):
        """
        check is this simple place of given coordinates is full (but also
        can be the final spot)
        """
        return False if self.rewards_table[x][y] == -1 else True

    def generate_random_start(self):
        """
        generate random coordinates to start
        """
        x = np.random.randint(self.nrows)
        y = np.random.randint(self.ncol)

        while self.maze[x][y] != FREE:
            x = np.random.randint(self.nrows)
            y = np.random.randint(self.ncol)
        return x, y

    def next_action(self, x, y):
        """
        chooses the best action to move
        but if epsilon is setted: randomize move sometimes
        """
        if self.epsilon is None:
            return np.argmax(self.q_values[x][y])
        else:
            if np.random.random() < self.epsilon:
                return np.argmax(self.q_values[x][y])
            else:
                return np.random.randint(4)

    def make_move(self, x, y,  action):
        """
        return new coordinates adter given action(move)
        """
        new_x, new_y = x, y
        if self.actions[action] == 'up' and x > 0:
            new_x -= 1
        elif self.actions[action] == 'right' and y < self.ncol - 1:
            new_y += 1
        elif self.actions[action] == 'down' and x < self.nrows - 1:
            new_x += 1
        elif self.actions[action] == 'left' and y > 0:
            new_y -= 1
        return new_x, new_y

    def train(self, start=None):
        """
        all magic happens here:
        start param:
        if it's None algorithm trains every episode from random location
        else it trains always from the same location
        """

        # lists to hold values for plots
        all_rewards = []
        len_paths = []
        for episode in range(self.episodes):
            x, y = start if start is not None else self.generate_random_start()
            all_rewards.append(0)
            path = []


            # while not self.is_illegal(x, y):
            while self.rewards_table[x][y] != 100:
                action = self.next_action(x, y)

                old_x, old_y = x, y
                x, y = self.make_move(old_x, old_y, action)

                path.append((x, y))

                reward = self.rewards_table[x][y]

                if self.rewards_table[x][y] != 100:
                    all_rewards[episode] += reward

                old_q_value = self.q_values[old_x, old_y, action]
                temp_diff = reward + (self.discount_factor * np.max(self.q_values[x, y])) - old_q_value

                new_q_value = old_q_value + (self.lr * temp_diff)
                self.q_values[old_x, old_y, action] = new_q_value

            len_paths.append(len(path))

        return all_rewards, len_paths

    def show_shortest_path(self, x, y):
        if self.is_illegal(x, y):
            raise IndexError("at this coordinates there is wall, we can't start from here")
        else:
            maze_copy = self.maze
            maze_copy[x][y] = 'S'
            shortest_path = [(x, y)]
            while not self.is_illegal(x, y):
                old_x, old_y = x, y
                action = self.next_action(x, y)

                x, y = self.make_move(x, y, action)
                if maze_copy[old_x][old_y] != 'S':
                    if self.actions[action] == 'up':
                        maze_copy[old_x][old_y] = '^'
                    elif self.actions[action] == 'right':
                        maze_copy[old_x][old_y] = '>'
                    elif self.actions[action] == 'down':
                        maze_copy[old_x][old_y] = 'v'
                    elif self.actions[action] == 'left':
                        maze_copy[old_x][old_y] = '<'

                shortest_path.append((x, y))
        return shortest_path, maze_copy


if __name__ == "__main__":
    # maze_path, goal, learing_rate, discount_factor (beta), episodes
    qmaze = Qlearning_maze("06-qlearning/maze1.txt", (10, 15), 0.8, 0.85, 50) 

    all_rewards, len_paths = qmaze.train((0,0))
    shortest_path, maze_with_path = qmaze.show_shortest_path(0, 0)
    print(shortest_path)
    print(maze_with_path)

    # plots creating
    plt.figure(0)
    plt.plot(all_rewards)
    plt.title('Sum of rewards obtained per episode')
    plt.xlabel('Episodes')
    plt.ylabel('Rewars')


    plt.figure(1)
    plt.plot(len_paths, 'red')
    plt.title('Sum of spots travelled per episode')
    plt.xlabel('Episodes')
    plt.ylabel('Spots')

    plt.show()
