import pygame
import numpy as np
import onnxruntime as ort
from env import DotsAndBoxesEnv
import os
import sys

# Claude
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)

class DotsAndBoxesGUI:
  def __init__(self, grid_size=5, model_path="models\dots_and_boxes_model.onnx"):
    self.grid_size = grid_size
    self.env = DotsAndBoxesEnv(grid_size=grid_size)

    try:
      model_path = resource_path(model_path)
      self.session = ort.InferenceSession(model_path)
      self.input_name = self.session.get_inputs()[0].name
      self.output_name = self.session.get_outputs()[0].name
      print(f"ONNX model loaded successfully from {model_path}")
    except Exception as e:
      print(f"Error loading ONNX model: {e}")
      sys.exit()

    pygame.init()

    self.BLACK = (0, 0, 0)
    self.WHITE = (255, 255, 255)
    self.RED = (255, 0, 0)
    self.BLUE = (0, 0, 255)
    self.GRAY = (128, 128, 128)

    self.cell_size = 100
    self.edge_thickness = 5
    self.dot_radius = 10
    self.margin = 100

    self.game_over = False
    self.human_player = "player_1"
    self.ai_player = "player_2"

    self.board_size = (self.grid_size - 1) * self.cell_size + 2 * self.margin

    self.width = self.board_size + 300

    self.screen = pygame.display.set_mode((self.width, self.board_size))
    pygame.display.set_caption("Dots and Boxes")

    self.font = pygame.font.Font(None, 36)
    self.small_font = pygame.font.Font(None, 24)

    self.reset()

  def reset(self):
    self.env.reset()
    self.game_over = False
    self.board = np.zeros(self.env.total_edges, dtype=np.int8)
    print("Game reset!")

  def edge_from_mouse(self, pos):
    x, y = pos
    x -= self.margin
    y -= self.margin

    if x < 0 or y < 0:
      return None

    for i in range(self.grid_size):
      for j in range(self.grid_size):
        edge_x = j * self.cell_size + self.cell_size // 2
        edge_y = i * self.cell_size

        if abs(x - edge_x) < self.cell_size // 2 and abs(y - edge_y) < self.edge_thickness * 2:
          return i * (self.grid_size - 1) + j

    for i in range(self.grid_size):
      for j in range(self.grid_size):
        edge_x = j * self.cell_size
        edge_y = i * self.cell_size + self.cell_size // 2

        if abs(x - edge_x) < self.edge_thickness * 2 and abs(y - edge_y) < self.cell_size // 2:
          return (self.grid_size - 1) * self.grid_size + i * self.grid_size + j

    return None

  def predict(self):
    obs = self.env.observe(self.ai_player)
    obs = obs.reshape((1, -1)).astype(np.int8)
    
    # Used Claude
    outputs = self.session.run(None, {self.input_name: obs})
    action_logits = outputs[0][0]
    valid_actions_mask = self.env.board_state == 0
    action_logits_masked = action_logits.copy()
    action_logits_masked[~valid_actions_mask] = -1e9
    action = np.argmax(action_logits_masked)
    
    return int(action)

  def draw_board(self):
    self.screen.fill(self.WHITE)

    board_state = self.board

    h_edges = board_state[:self.grid_size * (self.grid_size - 1)]
    v_edges = board_state[self.grid_size * (self.grid_size - 1):]

    for i in range(self.grid_size):
      for j in range(self.grid_size):
        x = j * self.cell_size + self.margin
        y = i * self.cell_size + self.margin
        pygame.draw.circle(self.screen, self.BLACK, (x, y), self.dot_radius)

    for i in range(self.grid_size):
      for j in range(self.grid_size - 1):
        edge_idx = i * (self.grid_size - 1) + j
        start_x = j * self.cell_size + self.margin + self.dot_radius
        start_y = i * self.cell_size + self.margin
        end_x = (j + 1) * self.cell_size + self.margin - self.dot_radius
        end_y = start_y
        if h_edges[edge_idx] == 1:
          pygame.draw.line(self.screen, self.BLUE, (start_x, start_y), (end_x, end_y), self.edge_thickness)
        elif h_edges[edge_idx] == 2:
          pygame.draw.line(self.screen, self.RED, (start_x, start_y), (end_x, end_y), self.edge_thickness)

    for i in range(self.grid_size - 1):
      for j in range(self.grid_size):
        edge_idx = i * self.grid_size + j
        start_x = j * self.cell_size + self.margin
        start_y = i * self.cell_size + self.margin + self.dot_radius
        end_x = start_x
        end_y = (i + 1) * self.cell_size + self.margin - self.dot_radius
        if v_edges[edge_idx] == 1:
          pygame.draw.line(self.screen, self.BLUE, (start_x, start_y), (end_x, end_y), self.edge_thickness)
        elif v_edges[edge_idx] == 2:
          pygame.draw.line(self.screen, self.RED, (start_x, start_y), (end_x, end_y), self.edge_thickness)

    claimed_boxes = self.env.claimed_boxes

    for i in range(self.grid_size - 1):
      for j in range(self.grid_size - 1):
        if claimed_boxes[i, j] != 0:
          x = j * self.cell_size + self.margin + self.dot_radius
          y = i * self.cell_size + self.margin + self.dot_radius
          size = (self.cell_size - 2 * self.dot_radius)

          if claimed_boxes[i, j] == 1:
            text = "H"
            pygame.draw.rect(self.screen, self.BLUE, (x, y, size, size))
          elif claimed_boxes[i, j] == 2:
            text = "A"
            pygame.draw.rect(self.screen, self.RED, (x, y, size, size))

          if text:
            text_surface = self.font.render(text, True, self.WHITE)
            text_rect = text_surface.get_rect(center=(x + size // 2, y + size // 2))
            self.screen.blit(text_surface, text_rect)

  def draw_info_panel(self):
    panel_x = self.board_size - 25
    panel_y = self.margin

    current = f"Current Player: {'Human' if self.env.agent_selection == self.human_player else 'AI'}"
    text = self.font.render(current, True, self.BLACK)
    self.screen.blit(text, (panel_x, panel_y))

    human_score = np.sum(self.env.claimed_boxes == 1)
    ai_score = np.sum(self.env.claimed_boxes == 2)

    score_text = f"Human Score: {human_score}"
    score_surface = self.font.render(score_text, True, self.BLUE)
    self.screen.blit(score_surface, (panel_x, panel_y + 40))

    score_text = f"AI Score: {ai_score}"
    score_surface = self.font.render(score_text, True, self.RED)
    self.screen.blit(score_surface, (panel_x, panel_y + 80))

    if self.game_over:
      if human_score > ai_score:
        winner_text = "Winner: Human"
      elif human_score < ai_score:
        winner_text = "Winner: AI"
      else:
        winner_text = "It's a draw!"

      winner_surface = self.font.render(winner_text, True, self.GRAY)
      self.screen.blit(winner_surface, (panel_x, panel_y + 120))

      restart = "Press R to Restart"
      restart_surface = self.small_font.render(restart, True, self.GRAY)
      self.screen.blit(restart_surface, (panel_x, panel_y + 160))

    instructions = [
      "Click on edges to place lines",
      "Complete boxes to score points",
      "Press R to restart game",
      "Press Q to quit",
    ]

    for i, instruction in enumerate(instructions):
      instruction_surface = self.small_font.render(instruction, True, self.GRAY)
      self.screen.blit(instruction_surface, (panel_x, panel_y + 200 + i * 30))

  def human_move(self, pos):
    edge_idx = self.edge_from_mouse(pos)
    if edge_idx is not None and self.env.board_state[edge_idx] == 0 and not self.game_over:
      self.env.step(edge_idx)
      self.board[edge_idx] = 1

      if self.env.terminations[self.human_player]:
        self.game_over = True
      return True
    return False

  def ai_move(self):
    action = self.predict()

    if self.env.board_state[action] == 0 and not self.game_over:
      self.env.step(action)
      self.board[action] = 2

      if self.env.terminations[self.ai_player]:
        self.game_over = True
      return True
    elif not self.game_over:
      actions = np.where(self.env.board_state == 0)[0]
      if actions.size > 0:
        action = np.random.choice(actions)
        self.env.step(action)
        self.board[action] = 2

        if self.env.terminations[self.ai_player]:
          self.game_over = True
      return True

    return False

  def run(self):
    clock = pygame.time.Clock()
    running = True

    while running:
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
          if not self.game_over and self.env.agent_selection == self.human_player:
            self.human_move(event.pos)

        elif event.type == pygame.KEYDOWN:
          if event.key == pygame.K_r:
            self.reset()
          elif event.key == pygame.K_q:
            running = False

      if not self.game_over and self.env.agent_selection == self.ai_player:
        pygame.time.delay(500)
        self.ai_move()

      self.draw_board()
      self.draw_info_panel()

      pygame.display.flip()
      clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
  gui = DotsAndBoxesGUI(grid_size=5, model_path="models\dots_and_boxes_model.onnx")
  gui.run()