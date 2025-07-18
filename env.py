from gymnasium.spaces import Box, Discrete
from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector
import numpy as np

class DotsAndBoxesEnv(AECEnv):
  metadata = {
    "render_modes": ["human"],
    "name": "dots_and_boxes",
    "is_parallelizable": True
  }

  def __init__(self, grid_size=5):
    super().__init__()
    self.grid_size = grid_size
    self.total_edges = 2 * (grid_size - 1) * grid_size
    self.action_space = Discrete(self.total_edges)
    self.agents = ["player_1", "player_2"]
    self.possible_agents = self.agents.copy()
    self.agent_selector = AgentSelector(self.agents)
    self.agent_selection = self.agent_selector.reset()
    self.render_mode = None

    obs_size = (
      self.total_edges +            # Edges
      (self.grid_size - 1) ** 2 +   # Box ownerships
      (self.grid_size - 1) ** 2 +   # Box completions counts
      self.total_edges +            # Action Priorities
      2 +                           # Player scores
      1                             # Turn
    )

    self.observation_spaces =  {
      "player_1": Box(low=0, high=1, shape=(obs_size,), dtype=np.float32),
      "player_2": Box(low=0, high=1, shape=(obs_size,), dtype=np.float32)
    }

    self.action_spaces = {
      "player_1": Discrete(self.total_edges),
      "player_2": Discrete(self.total_edges)
    }

  def observe(self, agent):
    if agent not in self.agents:
      return np.zeros(self.observation_spaces[agent].shape[0], dtype=np.int8)

    current = int(agent == self.agent_selection)

    box_edges = np.zeros((self.grid_size - 1, self.grid_size - 1), dtype=np.int8)
    
    for i in range(self.grid_size - 1):
      for j in range(self.grid_size - 1):
        edges = self._edges_for_box(i, j)
        box_edges[i, j] = np.sum(self.board_state[edge] for edge in edges)

    player_1_score = np.count_nonzero(self.claimed_boxes == 1)
    player_2_score = np.count_nonzero(self.claimed_boxes == 2)

    action_priorities = self._get_action_priorities()

    obs = np.concatenate([
      self.board_state.copy(),         # Edges
      self.claimed_boxes.flatten(),    # Box ownerships
      box_edges.flatten(),             # Box completions counts
      action_priorities,               # Action Priorities
      [player_1_score, player_2_score],# Player scores
      [current]                        # Turn
    ]).astype(np.int8)

  def _get_action_priorities(self):
    priorities = np.zeros(self.total_edges, dtype=np.int8)

    for action in range(self.total_edges):
      if self.board_state[action] == 1:
        priorities[action] = -2
        continue

      completing_boxes = 0
      three_edges = 0

      for i in range(self.grid_size - 1):
        for j in range(self.grid_size - 1):
          if self.claimed_boxes[i, j] == 0:
            edges = self._edges_for_box(i, j)
            if action in edges:
              current_edges = np.sum(self.board_state[edge] for edge in edges)
              if current_edges == 3:
                completing_boxes += 1
              elif current_edges == 2:
                three_edges += 1

      if completing_boxes > 0:
        priorities[action] = 2
      elif three_edges > 0:
        priorities[action] = -1
      else:
        priorities[action] = 1
    
    return priorities
        
  def observation_space(self, agent):
    return self.observation_spaces[agent]

  def action_space(self, agent):
    return self.action_spaces[agent]

  def reset(self, seed=None, options=None):
    self.agents = self.possible_agents.copy()
    self.agent_selection = self.agent_selector.reset()
    self.board_state = np.zeros(self.total_edges, dtype=np.int8)
    self.claimed_boxes = np.zeros((self.grid_size - 1, self.grid_size - 1), dtype=np.int8)

    self.rewards = { "player_1": 0, "player_2": 0 }
    self._cumulative_rewards = { "player_1": 0, "player_2": 0 }
    self.terminations = { "player_1": False, "player_2": False }
    self.truncations = { "player_1": False, "player_2": False }
    self.infos = { "player_1": {}, "player_2": {} }

    return {
      "player_1": self.observe("player_1"),
      "player_2": self.observe("player_2")
    }

  def step(self, action):
    agent = self.agent_selection

    if action is None:
      self._was_dead_step(action)
      self.agent_selection = self.agent_selector.next()
      return

    if self.board_state[action] == 1:
      self.rewards[agent] = -2
      self._accumulate_rewards()
      self.agent_selection = self.agent_selector.next()
      return

    boxes_before = self._get_box_edge_count()
    opponent = "player_1" if agent == "player_2" else "player_2"

    available_completions = self._count_available_completions(agent)

    self.board_state[action] = 1

    boxs_after = self._get_box_edge_count()
    reward = self._calculate_reward(agent, boxes_before, boxs_after, action)

    completed, temp_box_delta = _check_completed_boxes(agent)

    if completed > 0:
      reward += completed * 5
      self.claimed_boxes += temp_box_delta
    else:
      if available_completions > 0:
        created_three_edges = False
        for i in range(self.grid_size - 1):
          for j in range(self.grid_size - 1):
            if (boxes_before[i, j] == 2 and boxes_after[i, j] == 3 and self.claimed_boxes[i, j] == 0):
              created_three_edges = True
              break
        if created_three_edges:
          reward -= 5.0
        else:
          reward -= 2.0

      self.agent_selection = self.agent_selector.next()

    self.rewards[agent] = reward

    if self.board_state.sum() == self.total_edges:
      self.terminations = { "player_1": True, "player_2": True }
      
      player_1_boxes = np.count_nonzero(self.claimed_boxes == 1)
      player_2_boxes = np.count_nonzero(self.claimed_boxes == 2)

      if player_1_boxes > player_2_boxes:
        self.rewards["player_1"] += 10.0
        self.rewards["player_2"] -= 5.0
      elif player_2_boxes > player_1_boxes:
        self.rewards["player_2"] += 10.0
        self.rewards["player_1"] -= 5.0

    self._accumulate_rewards()

  def _check_completed_boxes(self, agent):
    agent_id = 1 if agent == "player_1" else 2
    completed = 0
    temp_box_delta = np.zeros_like(self.claimed_boxes)

    for i in range(self.grid_size - 1):
      for j in range(self.grid_size - 1):
        if self.claimed_boxes[i, j] == 0:
          edges = self._edges_for_box(i, j)
          if all(self.board_state[edge] == 1 for edge in edges):
            self.claimed_boxes[i, j] = agent_id
            completed += 1
            temp_box_delta[i, j] = 1

    return completed, temp_box_delta

  def _get_box_edge_count(self):
    box_counts = np.zeros((self.grid_size - 1, self.grid_size - 1), dtype=np.int8)
    for i in range(self.grid_size - 1):
      for j in range(self.grid_size - 1):
        edges = self._edges_for_box(i, j)
        box_counts[i, j] = np.sum(self.board_state[edge] for edge in edges)
    return box_counts

  def _count_available_completions(self, agent):
    count = 0
    for i in range(self.grid_size - 1):
      for j in range(self.grid_size - 1):
        if self.claimed_boxes[i, j] == 0:
          edges = self._edges_for_box(i, j)
          edge_counts = np.sum(self.board_state[edge] for edge in edges)
          if edge_counts == 3:
            count += 1
    return count

  def _calculate_reward(self, agent, boxes_before, boxes_after, action):
    reward = 0.05

    affected_boxes = []
    for i in range(self.grid_size - 1):
      for j in range(self.grid_size - 1):
        edges = self._edges_for_box(i, j)
        if action in edges:
          affected_boxes.append((i, j))

    three_edges_box_created = 0
    completing_box = False

    for i, j in affected_boxes:
      edges_before = boxes_before[i, j]
      edges_after = boxes_after[i, j]

      if edges_before == 2 and edges_after == 3 and self.claimed_boxes[i, j] == 0:
        three_edges_box_created += 1
        reward -= 3.0

      if edges_before == 3 and edges_after == 4:
        completing_box = True
        reward += 0.5
      elif edges_before == 1 and edges_after == 2 and self.claimed_boxes[i, j] == 0:
        reward -= 0.2
      elif edges_before == 0 and edges_after == 1:
        reward += 0.1

    if three_edges_box_created > 1:
      reward -= 2.0

    if completing_box:
      available_completions = self._count_available_completions(agent)
      if available_completions > 0:
        reward += 1.0
        
    return reward

  def _edges_for_box(self, row, col):
    top = row * (self.grid_size - 1) + col
    bottom = (row + 1) * (self.grid_size - 1) + col

    left = self.grid_size * (self.grid_size - 1) + row * self.grid_size + col
    right = left + 1

    return [top, bottom, left, right]

  def render(self, mode=None):
    size = self.grid_size
    h_edges = self.board_state[:size * (size - 1)].reshape((size, size - 1))
    v_edges = self.board_state[size * (size - 1):].reshape((size - 1, size))

    claimed = self.claimed_boxes.copy()
    string = ""

    for i in range(size):
      row = ""
      for j in range(size - 1):
        row += "•"
        row += "─" if h_edges[i, j] == 1 else " "
      row += "•\n"
      string += row

      if i < size - 1:
        row = ""
        for j in range(size):
          row += "│" if v_edges[i, j] == 1 else " "
          row += " " if claimed[i, j] == 0 else ("1" if claimed[i, j] == 1 else "2")
        row += "\n"
        string += row

    if self.render_mode == "human":
      print(string)