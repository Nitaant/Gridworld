import copy
import numpy as np
import random
import sys 
from collections import Counter
import time
import csv
import pandas as pd

def read_board(filename, reward):
    with open(filename) as file:
        board = [line.split() for line in file]
    
    row_size = len(board)
    col_size = len(board[0])
    Q = np.zeros((row_size,col_size))
    
    walls = []
    terminals = []

    for row in range(row_size):
        for col in range(col_size):
            if board[row][col] == "S":
                start_position = [row, col]
                Q[row][col] = reward    
            elif board[row][col] == "X":
                walls.append([row, col])
                Q[row][col] = reward
            elif board[row][col] == '0':
                Q[row][col] = reward
            else:
                Q[row][col] = board[row][col]
                terminals.append((row, col))

    RIGHT = np.zeros((row_size,col_size))
    LEFT = np.zeros((row_size,col_size))
    UP = np.zeros((row_size,col_size))
    DOWN = np.zeros((row_size,col_size))

    return start_position, row_size,col_size, walls, terminals, Q, LEFT, RIGHT, UP, DOWN, board



def takeAction(current_position, direction, walls, grid_row_size, grid_col_size, transition_value):
        new_position = copy.copy(current_position)
        X = transition_value
        Y = (1 - transition_value) / 2
        
        if direction == "up":
            action = np.random.choice(['up', '2up', 'down'], p=[X, Y, Y])
        if direction == "down":
            action = np.random.choice(['down', '2down', 'up'], p=[X, Y, Y])
        if direction == "left":
            action = np.random.choice(['left', '2left', 'right'], p=[X, Y, Y])
        if direction == "right":
            action = np.random.choice(['right', '2right', 'left'], p=[X, Y, Y])
        
        if action == 'up':
            new_position[0] += -1
        elif action == 'down':
            new_position[0] += 1
        elif action == 'right':
            new_position[1] += 1
        elif action == 'left':
            new_position[1] += -1
        
        elif action == '2up':
            new_position[0] += -1
            barrier = new_position in walls or new_position[0] < 0 or new_position[0] >= grid_row_size or new_position[1] < 0 or new_position[1] >= grid_col_size
            if not barrier:
                current_position = copy.copy(new_position)
                new_position[0] += -1
        
        elif action == '2down':
            new_position[0] += 1
            barrier = new_position in walls or new_position[0] < 0 or new_position[0] >= grid_row_size or new_position[1] < 0 or new_position[1] >= grid_col_size
            if not barrier:
                current_position = copy.copy(new_position)
                new_position[0] += 1
        
        elif action == '2right':
            new_position[1] += 1
            barrier = new_position in walls or new_position[0] < 0 or new_position[0] >= grid_row_size or new_position[1] < 0 or new_position[1] >= grid_col_size
            if not barrier:
                current_position = copy.copy(new_position)
                new_position[1] += 1
        
        elif action == '2left':
            new_position[1] += -1
            barrier = new_position in walls or new_position[0] < 0 or new_position[0] >= grid_row_size or new_position[1] < 0 or new_position[1] >= grid_col_size
            if not barrier:
                current_position = copy.copy(new_position)
                new_position[1] += -1

        barrier = new_position in walls or new_position[0] < 0 or new_position[0] >= grid_row_size or new_position[1] < 0 or new_position[1] >= grid_col_size
        if not barrier:
            current_position = new_position

        return current_position


def create_episode(current_position, terminals, step_size, gamma, Q, UP, DOWN, LEFT, RIGHT, transition_value, epsilon, walls):
    remember_route = []
    episode_cost = 0  
    
    while Q[current_position[0], current_position[1]] <= 0 or tuple([current_position[0], current_position[1]]) not in terminals:
        #and tuple(current_position[0], current_position[1]) not in terminals
        
        if random.uniform(0,1) > epsilon:
            max_direction = np.argmax([UP[current_position[0], current_position[1]], DOWN[current_position[0], current_position[1]], LEFT[current_position[0], current_position[1]], RIGHT[current_position[0], current_position[1]]])
            direction = ["up", "down", "left", "right"][max_direction]
        else:   
            direction = np.random.choice(["up", "down", "left", "right"])
        
        new_position = takeAction(current_position, direction, walls, row_size,col_size, transition_value)

        if tuple([new_position[0], new_position[1]]) in terminals:   
            Q_max = Q[new_position[0], new_position[1]]

            if direction == "up":
               q_value = UP[current_position[0], current_position[1]] + step_size * (Q[current_position[0], current_position[1]] + gamma * (Q_max - UP[current_position[0], current_position[1]]))
               UP[current_position[0], current_position[1]] = q_value
               episode_cost += q_value

            elif direction == "down":
               q_value = DOWN[current_position[0], current_position[1]] + step_size * (Q[current_position[0], current_position[1]] + gamma * (Q_max - DOWN[current_position[0], current_position[1]]))
               DOWN[current_position[0], current_position[1]] = q_value
               episode_cost += q_value

            elif direction == "left":
               q_value = LEFT[current_position[0], current_position[1]] + step_size * (Q[current_position[0], current_position[1]] + gamma * (Q_max - LEFT[current_position[0], current_position[1]]))
               LEFT[current_position[0], current_position[1]] = q_value
               episode_cost += q_value

            elif direction == "right":
               q_value = RIGHT[current_position[0], current_position[1]] + step_size * (Q[current_position[0], current_position[1]] + gamma * (Q_max - RIGHT[current_position[0], current_position[1]]))
               RIGHT[current_position[0], current_position[1]] = q_value
               episode_cost += q_value

               remember_route.append(current_position)

            return Q, UP, DOWN, LEFT, RIGHT, remember_route, episode_cost
                

        Q_max = max(UP[new_position[0], new_position[1]], DOWN[new_position[0], new_position[1]], LEFT[new_position[0], new_position[1]], RIGHT[new_position[0], new_position[1]])
        
        if direction == "up":
            q_value = UP[current_position[0], current_position[1]] + step_size * (Q[current_position[0], current_position[1]] + gamma * (Q_max - UP[current_position[0], current_position[1]]))
            UP[current_position[0], current_position[1]] = q_value
            episode_cost += Q[new_position[0], new_position[1]]
        
        elif direction == "down":
            q_value = DOWN[current_position[0], current_position[1]] + step_size * (Q[current_position[0], current_position[1]] + gamma * (Q_max - DOWN[current_position[0], current_position[1]]))
            DOWN[current_position[0], current_position[1]] = q_value
            episode_cost += Q[new_position[0], new_position[1]]

        elif direction == "left":
            q_value = LEFT[current_position[0], current_position[1]] + step_size * (Q[current_position[0], current_position[1]] + gamma * (Q_max - LEFT[current_position[0], current_position[1]]))
            LEFT[current_position[0], current_position[1]] = q_value
            episode_cost += Q[new_position[0], new_position[1]]

        elif direction == "right":
            q_value = RIGHT[current_position[0], current_position[1]] + step_size * (Q[current_position[0], current_position[1]] + gamma * (Q_max - RIGHT[current_position[0], current_position[1]]))
            RIGHT[current_position[0], current_position[1]] = q_value
            episode_cost += Q[new_position[0], new_position[1]]

        current_position = copy.copy(new_position)
        remember_route.append(current_position)
    

    return Q, UP, DOWN, LEFT, RIGHT, remember_route, episode_cost


def display_policy(Q, UP, DOWN, LEFT, RIGHT, row_size,col_size, start_position, board):
    
    print("------- POLICY MAP ------", end="\n\n")

    type_arrow = {0:"^", 1:"V", 2:"<", 3:">"}
    for i in range(row_size):
        for j in range(col_size):
            if board[i][j] == "X":
                print("X", end="\t")
            elif [i, j] == start_position:
                arrow = np.argmax([UP[i,j], DOWN[i,j], LEFT[i,j], RIGHT[i,j]])
                print("{}S".format(type_arrow[arrow]), end="\t")
            elif Q[i][j] == "X":
                print("X", end="\t")
            elif Q[i][j] == "K":
                print("K", end="\t")
            elif Q[i][j] >= 1 or Q[i][j] < 0 and Q[i][j] != reward:
                print(Q[i][j], end="\t")
            else:
                arrow = np.argmax([UP[i,j], DOWN[i,j], LEFT[i,j], RIGHT[i,j]])
                print(type_arrow[arrow], end="\t")
        print("\n")


def display_heat_map(tuple_heat_map, Q, start_position, total_time_spent):
    
    print("-------- HEAT MAP ------", end="\n\n")

    for i in range(len(Q)):
        for j in range(len(Q[i])):
            if Q[i][j] >= 1 or Q[i][j] < 0 and Q[i][j] != reward:
                print(round(Q[i][j], 3), end="\t")
            elif Q[i][j] == "X":
                print("X", end="\t")
            #elif [i,j] == start_position:
                #print("S", end="\t")
            else:
                t_p = ((tuple_heat_map[i, j] / total_time_spent)) * 100
                print("{}%".format(round(t_p, 2)), end="\t")
        print("\n")


if __name__ == "__main__":  
    arguements = sys.argv[1:]
    filename = arguements[0]
    reward = float(arguements[1])
    gamma = float(arguements[2])
    run_time = float(arguements[3])
    step_size = 0.1
    transition_value = float(arguements[4])

    if run_time <= 0:
        print("RUNTIME CANNOT BE LESS THAN OR EQUAL TO 0")
        exit()

    
    current_position, row_size,col_size, walls, terminals, Q, LEFT, RIGHT, UP, DOWN, board = read_board(filename, reward)   
    starting_position = copy.copy(current_position)
    n = 0
    total_cost = 0
    heat_map = []
    start = time.time()
    record_cost = []
    batch_cost = []
    epsilon = 1
    factor = 0.9
    record_time = time.time()
    while time.time() - start < run_time:
        Q, UP, DOWN, LEFT, RIGHT, remember_route, episode_cost = create_episode(starting_position, terminals, step_size, gamma, Q, UP, DOWN, LEFT, RIGHT, transition_value, epsilon, walls)
        batch_cost.append(episode_cost)

        if time.time() - record_time > 0.1:
            record_cost.append(sum(batch_cost)/len(batch_cost))

        if time.time() - record_time > 0.01:
            epsilon = factor * epsilon
            #print(epsilon)

        total_cost += episode_cost
        heat_map.extend(remember_route)
        n += 1

    df = pd.DataFrame(record_cost);
    df.to_csv('big10second001.csv',index=False)

    tuple_heat_map = Counter([ (loc[0], loc[1]) for loc in heat_map]) 

    total_time_spent = sum(tuple_heat_map.values())

    display_policy(Q, UP, DOWN, LEFT, RIGHT, row_size,col_size, starting_position, board)

    print("\n\n")

    display_heat_map(tuple_heat_map, Q, starting_position, total_time_spent)

    print("\n")

    print("reward:{} gamma:{} runtime:{} transition value:{}".format(arguements[1], arguements[2], arguements[3], arguements[4]))

    print("\n")

    print("average reward per episode: {}".format(total_cost/n))

    print("press Q to exit")

    #for i in range(len(Q)):
     #   for j in range(len(Q[0])):
      #      if [i,j] == starting_position:
       #         print("S", end="   ")
        #    else:
         #       print(Q[i][j], end='   ')
        #print("")

    from matplotlib import pyplot as plt
    import cv2
    
    plt.plot(range(len(record_cost)), record_cost)
    plt.title("epsilon decay by a factor of 0.9 every 0.01 seconds (epsilon starts at 1)")
    plt.xlabel("EPISODE BATCH NUMBER", fontsize=10)
    plt.ylabel("AVERAGE REWARD", fontsize=10)
    plt.legend(title="average reward per episode {}".format(round((total_cost/n), 5)))
    plt.show()

    # if cv2.waitKey() == "q":
    #     cv2.destroyAllWindows()
