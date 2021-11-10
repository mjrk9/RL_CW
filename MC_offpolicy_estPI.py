# This class define the Monte-Carlo agent

class MC_agent(object):

  # [Action required]
  # WARNING: make sure this function can be called by the auto-marking script

  def get_action(self, state_policy, epsilon):
    max_val = np.argmax(state_policy)
    rand_val = np.random.uniform(0,1)
    if rand_val < (1-epsilon):
      return max_val
    else:
      options = []
      for idx in range(len(state_policy)):
        if idx != max_val:
          options.append(idx)
      return np.random.choice(options)


  def solve(self, env):
    """
    Solve a given Maze environment using Monte Carlo learning
    input: env {Maze object} -- Maze to solve
    output: 
      - policy {np.array} -- Optimal policy found to solve the given Maze environment 
      - values {list of np.array} -- List of successive value functions for each episode 
      - total_rewards {list of float} -- Corresponding list of successive total non-discounted sum of reward for each episode 
    """

    # Initialisation (can be edited)
    epsilon = 0.2
    gamma = env.get_gamma()
    Q = np.random.rand(env.get_state_size(), env.get_action_size()) 
    C = np.zeros((env.get_state_size(), env.get_action_size())) 
    V = np.zeros(env.get_state_size())
    total_rewards = [[[] for i in range(env.get_action_size())] for j in range(env.get_state_size())]

    policy = np.argmax(Q, axis=1)

    num_runs = 50
    run = 0
    while run < num_runs:
      run += 1

      time, env_state, reward, Terminate = env.reset()
      b = np.zeros((env.get_state_size(), env.get_action_size())) + 0.25

      S_arr = np.array([], dtype = int)
      S_arr = np.append(S_arr, env_state)
      
      A_arr = np.array([], dtype = int)
      R_arr = np.array([], dtype = int)
      
      while not Terminate:
        action = self.get_action(b[env_state], epsilon)
        A_arr = np.append(A_arr, action)
        time, env_state, reward, Terminate = env.step(action)
         
        S_arr = np.append(S_arr, env_state)
        R_arr = np.append(R_arr, reward) 

      G = 0
      W = 1
      T = time-1
      
      for t in range(T-1, -1, -1):
        St = S_arr[t]; At = A_arr[t]
        # G = G + R_arr[t+1]
        G = (G * gamma) + R_arr[t+1]
        total_rewards[St][At].append(G)

        Q[St][At] = Q[St][At] + ((W / C[St][At]) * (G-Q[St][At]))
        policy[St] = np.argmax(Q[St])
        if At != policy[St]:
          continue
        W = W * (1 / b[St][At])



    for j in range(len(Q)):
      V[j] = np.max(Q[j])
    values = [V]
    #### 
    # Add your code here
    # WARNING: this agent only has access to env.reset() and env.step()
    # You should not use env.get_T(), env.get_R() or env.get_absorbing() to compute any value
    ####
    #def step()
    
    return policy, values, total_rewards