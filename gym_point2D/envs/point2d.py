import numpy as np
import gym
from gym import spaces
import random
import matplotlib.pyplot as plt


class Point2DEnv(gym.Env):
  """
  Custom Environment that follows gym interface.
  """
  # Because of google colab, we cannot implement the GUI ('human' render mode)
  metadata = {'render.modes': ['human']}
  # Define constants for clearer code


  def __init__(self, maxX = 1.0, maxY = 1.0, grid_size=10):
    super(Point2DEnv, self).__init__()
    self.eps = 0.1
    self.s = 0.1
    self.t = 0.1
    self.minaccX = -1.0
    self.minaccY = -1.0
    self.maxaccX = 1.0
    self.maxaccY = 1.0

    self.minvelX = -1.0
    self.minvelY = -1.0
    self.maxvelX = 1.0
    self.maxvelY = 1.0
    # Size of the 1D-grid
    # self.grid_size = grid_size
    self.grid_width = maxX
    self.grid_height = maxY
    # Initialize the agent at the right of the grid
    # self.agent_pos = grid_size - 1
    self.agent_state = np.array([0.0,0.0,0.0,0.0])
    self.agent_state[0] = np.random.uniform(0.0,maxX)
    self.agent_state[1] = np.random.uniform(0.0,maxY)
    self.agent_state[2] = np.random.uniform(self.minvelX,self.maxvelX)
    self.agent_state[3] = np.random.uniform(self.minvelY,self.maxvelY)

    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions, we have two: left and right
    # n_actions = 2
    # self.action_space = spaces.Discrete(n_actions)
    self.action_space = spaces.Box(low=np.array([self.minaccX,self.minaccY]), high=np.array([self.maxaccX,self.maxaccY]), dtype=np.float32)
    # The observation will be the coordinate of the agent
    # this can be described both by Discrete and Box space
    self.observation_space = spaces.Box(low=np.array([0.0,0.0,self.minvelX,self.minvelY]), high=np.array([self.grid_width,self.grid_height,self.maxvelX,self.maxvelY]),
                                        dtype=np.float32)
  def outofbounds(self):
    if (self.agent_state[0] > self.grid_width) or (self.agent_state[0] < 0.0):
      return True
    elif (self.agent_state[1] > self.grid_height) or (self.agent_state[1] < 0.0):
      return True
    return False

  def dist(self):
    return np.sqrt((self.grid_width - self.agent_state[0])**2 + (self.grid_height - self.agent_state[1])**2)


  def reset(self):
    """
    Important: the observation must be a numpy array
    :return: (np.array) 
    """
    # Initialize the agent at the right of the grid
    # self.agent_pos = self.grid_size - 1
    # self.agent_state = np.array([0.0,0.0,0.0,0.0])
    self.agent_state[0] = np.random.uniform(0.0,self.grid_width)
    self.agent_state[1] = np.random.uniform(0.0,self.grid_height)
    self.agent_state[2] = np.random.uniform(self.minvelX,self.maxvelX)
    self.agent_state[3] = np.random.uniform(self.minvelY,self.maxvelY)
    return self.agent_state

  def step(self, action):
    # if action == self.LEFT:
    #   self.agent_pos -= 1
    # elif action == self.RIGHT:
    #   self.agent_pos += 1
    # else:
    #   raise ValueError("Received invalid action={} which is not part of the action space".format(action))
    a_x = action[0]
    a_y = action[1]
    
    #Change in pos
    meanX = self.agent_state[0] + self.agent_state[2]*self.t + 0.5*a_x*(self.t**2)
    meanY = self.agent_state[1] + self.agent_state[3]*self.t + 0.5*a_y*(self.t**2)
    
    #Change in vel
    meanvX = self.agent_state[2] + a_x*(self.t)
    meanvY = self.agent_state[3] + a_y*(self.t)

    #Stochastic transition
    mu = np.array([meanX,meanY,meanvX,meanvY])
    sigma = (self.s**2)*np.eye(4)
    self.agent_state = np.random.multivariate_normal(mu, sigma)

    # Account for the boundaries of the grid
    # self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size)
    if self.outofbounds() == True:
      done = True
      reward = -1000
      self.agent_state = self.reset()
    else:
      done = bool(np.abs(self.agent_state[0] - self.grid_width) < self.eps and np.abs(self.agent_state[1] - self.grid_height) < self.eps)
      reward = 1.0/(self.dist()+self.eps)
      if done == True :
        reward += 1000.0


    # Are we at the left of the grid?
    # done = bool(self.agent_pos == 0)

    # Null reward everywhere except when reaching the goal (left of the grid)
    # reward = 1 if self.agent_pos == 0 else 0

    # Optionally we can pass additional info, we are not using that for now
    info = {}

    return self.agent_state, reward, done, info

  def render(self, mode='console'):
    # if mode != 'console':
    #   raise NotImplementedError()
    # agent is represented as a cross, rest as a dot
    # print("." * self.agent_pos, end="")
    # print("x", end="")
    # print("." * (self.grid_size - self.agent_pos))
    plt.scatter([self.agent_state[0],self.grid_width],[self.agent_state[1],self.grid_height])
    plt.show()

  def close(self):
    pass
    



# env = Point2D()

# obs = env.reset()
# env.render()

# print(env.observation_space)
# print(env.action_space)
# print(env.action_space.sample())

# ACTION = np.array([1,1])
# # Hardcoded best agent: always go left!
# n_steps = 20
# for step in range(n_steps):
#   print("Step {}".format(step + 1))
#   obs, reward, done, info = env.step(ACTION)
#   print('obs=', obs, 'reward=', reward, 'done=', done)
#   env.render()
#   if done:
#     print("Goal reached!", "reward=", reward)
#     break
