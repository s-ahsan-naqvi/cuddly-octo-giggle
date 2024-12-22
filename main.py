import pygame
import numpy as np
import random
import time
import csv

pygame.init()

WIDTH, HEIGHT = 600, 600
ROWS, COLS = 10, 10
CELL_SIZE = WIDTH // COLS

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

REWARD_GOAL = 10
PENALTY_OBSTACLE = -10
STEP_COST = -1

DISCOUNT_FACTOR = 0.9
EPSILON = 0.1
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("MDP Pathfinding Simulation")

env = np.zeros((ROWS, COLS))
start = (0, 0)
goal = (ROWS - 1, COLS - 1)
obstacles = [(3, 3), (3, 4), (4, 3), (6, 7), (1, 5), (7, 2), (5, 5), (5, 0), (5, 4), (5, 3), (5, 2), (5, 1), (5, 0), (0,5)]

for obs in obstacles:
    env[obs] = PENALTY_OBSTACLE

env[goal] = REWARD_GOAL

def draw_grid(values=None, policy=None):
    for row in range(ROWS):
        for col in range(COLS):
            rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if (row, col) == start:
                pygame.draw.rect(screen, BLUE, rect)
            elif (row, col) == goal:
                pygame.draw.rect(screen, GREEN, rect)
            elif (row, col) in obstacles:
                pygame.draw.rect(screen, RED, rect)
            else:
                pygame.draw.rect(screen, WHITE, rect)
                pygame.draw.rect(screen, BLACK, rect, 1)
            
            if values is not None and policy is not None:
                font = pygame.font.Font(None, 20)
                value_text = font.render(f"{values[row, col]:.1f}", True, YELLOW)
                screen.blit(value_text, (col * CELL_SIZE + 5, row * CELL_SIZE + 5))
                if policy[row, col]:
                    arrow = {'UP': '^', 'DOWN': 'v', 'LEFT': '<', 'RIGHT': '>'}.get(policy[row, col], '')
                    arrow_text = font.render(arrow, True, BLACK)
                    screen.blit(arrow_text, (col * CELL_SIZE + CELL_SIZE // 2 - 5, row * CELL_SIZE + CELL_SIZE // 2 - 5))

def get_next_state(state, action):
    row, col = state
    if action == 'UP':
        row = max(0, row - 1)
    elif action == 'DOWN':
        row = min(ROWS - 1, row + 1)
    elif action == 'LEFT':
        col = max(0, col - 1)
    elif action == 'RIGHT':
        col = min(COLS - 1, col + 1)
    return (row, col)

def run_mdp():
    values = np.zeros((ROWS, COLS))
    policy = np.full((ROWS, COLS), '', dtype=object)

    with open('mdp_iterations.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Iteration", "Row", "Col", "Value", "Policy"])

        for iteration in range(500):  
            new_values = np.copy(values)

            for row in range(ROWS):
                for col in range(COLS):
                    if (row, col) == goal or (row, col) in obstacles:
                        continue

                    max_value = float('-inf')
                    best_action = None

                    for action in ACTIONS:
                        next_state = get_next_state((row, col), action)
                        reward = env[next_state] + STEP_COST
                        value = reward + DISCOUNT_FACTOR * values[next_state]

                        if value > max_value:
                            max_value = value
                            best_action = action

                    new_values[row, col] = max_value
                    policy[row, col] = best_action

                    writer.writerow([iteration, row, col, max_value, best_action])
                    print([iteration, row, col, max_value, best_action])

            values = new_values

            if iteration % 50 == 0: 
                screen.fill(BLACK)
                draw_grid(values, policy)
                pygame.display.flip()
                time.sleep(0.5)  

    return values, policy

def simulate(values, policy):
    state = start
    clock = pygame.time.Clock()

    while state != goal:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        action = policy[state]
        state = get_next_state(state, action)

        screen.fill(BLACK)
        draw_grid(values, policy)
        rect = pygame.Rect(state[1] * CELL_SIZE, state[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, BLUE, rect)
        pygame.display.flip()

        clock.tick(5)

values, policy = run_mdp()
simulate(values, policy)
pygame.quit()
