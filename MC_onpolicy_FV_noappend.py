# This class define the Monte-Carlo agent

class MC_agent(object):

  # [Action required]
  # WARNING: make sure this function can be called by the auto-marking script

  def get_action(self, Q_state, epsilon):
    max_val = np.argmax(Q_state)
    rand_val = np.random.uniform(0,1)
    if rand_val < (1-epsilon):
      return max_val
    else:
      options = []
      for idx in range(len(Q_state)):
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
    epsilon = 0.05
    alpha = 0.1
    Q = np.random.rand(env.get_state_size(), env.get_action_size()) 
    V = np.zeros(env.get_state_size())
    gamma = env.get_gamma()
    policy = np.zeros((env.get_state_size(), env.get_action_size())) 
    # policy[:, 0] = np.argmax(Q, axis=1)
    for state in range(len(Q)):
      policy[state][np.argmax(Q[state])] = 1

    values = [V]

    # total_rewards = [[[] for i in range(env.get_action_size())] for j in range(env.get_state_size())]
    total_rewards = [[[0,0] for i in range(env.get_action_size())] for j in range(env.get_state_size())]

    num_runs = 10000
    run = 0
    while run < num_runs:
      visited = {}
      run += 1
      time, env_state, reward, Terminate = env.reset()
      S_arr = np.array([], dtype = int)      
      A_arr = np.array([], dtype = int)
      R_arr = np.array([], dtype = int)
      R_arr = np.append(R_arr, 0)

      while not Terminate:
        action = self.get_action(Q[env_state], epsilon)
        S_arr = np.append(S_arr, env_state)
        A_arr = np.append(A_arr, action)
        sa_pair = (env_state, action) 
        if sa_pair not in visited:
          visited[sa_pair] = time

        time, env_state, reward, Terminate = env.step(action)
        R_arr = np.append(R_arr, reward) 

      G = 0
      T = len(S_arr)

      for t in range(T-1, -1, -1):

        St = S_arr[t]; At = A_arr[t]
        # G = G + R_arr[t+1]
        G = (G * gamma) + R_arr[t+1]

        if t > visited[(St, At)]:
          total_rewards[St][At][0] += 1
          # total_rewards[St][At][1] = total_rewards[St][At][1] + ((G - total_rewards[St][At][1]) / total_rewards[St][At][0])
          total_rewards[St][At][1] = total_rewards[St][At][1] + ((G - total_rewards[St][At][1]) * alpha)

          # Q[St][At] = np.mean(total_rewards[St][At])
          Q[St][At] = total_rewards[St][At][1]

          if St == 12:
            print("Q val is ", Q[St])

          best_action = np.argmax(Q[St])
          V[St] = np.max(Q[St])
          for a in range(len(Q[St])):
            if a == best_action:
              policy[St][a] = 1 - epsilon + (epsilon/len(Q[St]))
            else:
              policy[St][a] = (epsilon/len(Q[St]))

    # for j in range(len(Q)):
    #   V[j] = np.max(Q[j])
    V = np.einsum('sa,sa->s',policy,Q)
    values = [V]
    
    #### 
    # Add your code here
    # WARNING: this agent only has access to env.reset() and env.step()
    # You should not use env.get_T(), env.get_R() or env.get_absorbing() to compute any value
    ####
    #def step()
    
    return policy, values, total_rewards