import pygame
from env import DotsAndBoxesEnv
from stable_baselines3 import PPO
import numpy as np

env = DotsAndBoxesEnv()

colors = {
  'BLACK': (0, 0, 0),
  'WHITE': (255, 255, 255),
  'RED': (255, 0, 0),
  'GREEN': (0, 255, 0),
  'BLUE': (0, 0, 255),
  'YELLOW': (255, 255, 0),
  'GRAY': (128, 128, 128),
  'LIGHT_GRAY': (200, 200, 200),
  'DARK_GRAY': (64, 64, 64),
}

settings = {
  'grid_size': 5,
  'cell_size': 100,
  'edge_thickness': 5,
  'dot_radius': 10,
  'margin': 100,
}

game_over = False
human_player = "player_1"
ai_player = "player_2"
winner = None

board_size = (settings['grid_size'] - 1) * settings['cell_size'] + 2 * settings['margin']
width = board_size + 300

try:
  model = PPO.load("models/dots_and_boxes_model")
except Exception as e:
  print(f"Error loading model: {e}")
  exit()

pygame.init()

screen = pygame.display.set_mode((width, board_size))
pygame.display.set_caption("Dots and Boxes")

font = pygame.font.Font(None, 36)
small_font = pygame.font.Font(None, 24)

def reset():
  global env, game_over, winner

  env.reset()
  game_over = False
  winner = None

def edge_from_mouse(pos):
  global settings

  x, y = pos
  x -= settings['margin']
  y -= settings['margin']

  if x < 0 or y < 0:
    return None

  for i in range(settings['grid_size']):
    for j in range(settings['grid_size']):
      edge_x = j * settings['cell_size'] + settings['cell_size'] // 2
      edge_y = i * settings['cell_size']

      if (abs(x - edge_x) < settings['cell_size'] // 2 and
          abs(y - edge_y) < settings['edge_thickness']):
        return i * (settings['grid_size'] - 1) + j

  for i in range(settings['grid_size']):
    for j in range(settings['grid_size']):
      edge_x = j * settings['cell_size']
      edge_y = i * settings['cell_size'] + settings['cell_size'] // 2

      if (abs(x - edge_x) < settings['edge_thickness'] and
          abs(y - edge_y) < settings['cell_size'] // 2):
        return (settings['grid_size'] - 1) * settings['grid_size'] + i * settings['grid_size'] + j

  return None

def predict():
  global model, env, ai_player

  obs = env.observe(ai_player)
  obs = obs.reshape(1, -1)
  action, _ = model.predict(obs, deterministic=True)
  return action

def draw_board():
  global screen, colors, settings, env, font, small_font

  screen.fill(colors['WHITE'])

  for i in range(settings['grid_size']):
    for j in range(settings['grid_size']):
      x = j * settings['cell_size'] + settings['margin']
      y = i * settings['cell_size'] + settings['margin']
      pygame.draw.circle(screen, colors['BLACK'], (x, y), settings['dot_radius'])

  board_state = env.board_state

  h_edges = board_state[:settings['grid_size'] * (settings['grid_size'] - 1)]

  for i in range(settings['grid_size']):
    for j in range(settings['grid_size'] - 1):
      edge_idx = i * (settings['grid_size'] - 1) + j
      if h_edges[edge_idx] == 1:
        start_x = j * settings['cell_size'] + settings['margin'] + settings['dot_radius']
        start_y = i * settings['cell_size'] + settings['margin']
        end_x = (j + 1) * settings['cell_size'] + settings['margin'] - settings['dot_radius']
        end_y = start_y

        pygame.draw.line(screen, colors['BLACK'], (start_x, start_y), (end_x, end_y), settings['edge_thickness'])

  v_edges = board_state[settings['grid_size'] * (settings['grid_size'] - 1):]

  for i in range(settings['grid_size'] - 1):
    for j in range(settings['grid_size']):
      edge_idx = i * settings['grid_size'] + j
      if v_edges[edge_idx] == 1:
        start_x = j * settings['cell_size'] + settings['margin']
        start_y = i * settings['cell_size'] + settings['margin'] + settings['dot_radius']
        end_x = start_x
        end_y = (i + 1) * settings['cell_size'] + settings['margin'] - settings['dot_radius']

        pygame.draw.line(screen, colors['BLACK'], (start_x, start_y), (end_x, end_y), settings['edge_thickness'])

  claimed_boxes = env.claimed_boxes

  for i in range(settings['grid_size'] - 1):
    for j in range(settings['grid_size'] - 1):
      if claimed_boxes[i, j] != 0:
        x = j * settings['cell_size'] + settings['margin'] + settings['dot_radius']
        y = i * settings['cell_size'] + settings['margin'] + settings['dot_radius']
        size = (settings['cell_size'] - 2 * settings['dot_radius'])

        if claimed_boxes[i, j] == 1:
          pygame.draw.rect(screen, colors['RED'], (x, y, size, size))
          text = "H"
        elif claimed_boxes[i, j] == 2:
          pygame.draw.rect(screen, colors['BLUE'], (x, y, size, size))
          text = "A"

        if text:
          text_surface = font.render(text, True, colors['WHITE'])
          text_rect = text_surface.get_rect(center=(x + size // 2, y + size // 2))
          screen.blit(text_surface, text_rect)

def draw_info_panel():
  global screen, colors, settings, env, font, small_font, human_player, game_over

  panel_x = settings['margin'] + (settings['grid_size'] - 1) * settings['cell_size'] + 50
  panel_y = settings['margin']

  current = f"Current Player: {'Human' if env.agent_selection == human_player else 'AI'}"
  text = font.render(current, True, colors['BLACK'])
  screen.blit(text, (panel_x, panel_y))

  human_score = np.sum(env.claimed_boxes == 1)
  ai_score = np.sum(env.claimed_boxes == 2)

  score = f"Human: {human_score}"
  text = font.render(score, True, colors['RED'])
  screen.blit(text, (panel_x, panel_y + 40))

  score = f"AI: {ai_score}"
  text = font.render(score, True, colors['BLUE'])
  screen.blit(text, (panel_x, panel_y + 80))

  if game_over:
    if human_score > ai_score:
      winner_text = "Human wins!"
      color = colors['RED']
    elif ai_score > human_score:
      winner_text = "AI wins!"
      color = colors['BLUE']
    else:
      winner_text = "It's a draw!"
      color = colors['YELLOW']

    text = font.render(winner_text, True, color)
    screen.blit(text, (panel_x, panel_y + 120))

    restart_text = "Press R to restart"
    text = small_font.render(restart_text, True, colors['GRAY'])
    screen.blit(text, (panel_x, panel_y + 160))

  instructions = [
    "Click on edges to place lines",
    "Complete boxes to score points",
    "Press R to restart game",
    "Press Q to quit",
  ]
  
  for i, instruction in enumerate(instructions):
    text = small_font.render(instruction, True, colors['GRAY'])
    screen.blit(text, (panel_x, panel_y + 200 + i * 30))

def human_move(edge_idx):
  global env, human_player, game_over

  if edge_idx is not None and env.board_state[edge_idx] == 0 and not game_over:
    res = env.step(edge_idx)

    if env.terminations[human_player]:
      game_over = True

    return True
  return False

def ai_move():
  global env, ai_player, game_over

  action = predict()

  if env.board_state[action] == 0 and not game_over:
    res = env.step(action)
    if env.terminations[ai_player]:
      game_over = True
    return True
  else:
    valid_actions = np.where(env.board_state == 0)[0]
    if valid_actions.size > 0:
      res = env.step(valid_actions[0])
      if env.terminations[ai_player]:
        game_over = True
      return True

  return False

def run():
  global screen, game_over, human_player, ai_player, winner, env
  clock = pygame.time.Clock()
  running = True

  while running:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False
      
      elif event.type == pygame.KEYDOWN:
        if event.key == pygame.K_r:
          reset()
        if event.key == pygame.K_q:
          running = False

      elif event.type == pygame.MOUSEBUTTONDOWN:
        if not game_over and env.agent_selection == human_player:
          edge_idx = edge_from_mouse(event.pos)
          human_move(edge_idx)

    if not game_over and env.agent_selection == ai_player:
      pygame.time.delay(500)  # Delay for AI move
      ai_move()

    draw_board()
    draw_info_panel()

    pygame.display.flip()
    clock.tick(60)

  pygame.quit()
  exit()

if __name__ == "__main__":
  reset()
  run()