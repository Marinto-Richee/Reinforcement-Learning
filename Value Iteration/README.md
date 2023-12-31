# VALUE ITERATION ALGORITHM

## AIM
The aim of this experiment is to implement and understand the Value Iteration algorithm, a fundamental concept in reinforcement learning.

## PROBLEM STATEMENT
The problem statement involves the FrozenLake environment in OpenAI Gym, which is a gridworld problem challenging reinforcement learning agents to navigate a slippery terrain to reach a goal state while avoiding hazards. The environment is closed with a fence, so the agent cannot leave the gridworld.

### States
- **5 Terminal States:**
  - G (Goal): The state the agent aims to reach.
  - H (Hole): A hazardous state that the agent must avoid at all costs.
- **11 Non-terminal States:**
  - S (Starting state): The initial position of the agent.
  - Intermediate states: Grid cells forming a layout that the agent must traverse.

### Actions
The agent can take 4 actions in each state:
- LEFT
- RIGHT
- UP
- DOWN

### Transition Probabilities
The environment is stochastic, meaning that the outcome of an action is not always certain.
- 33.33% chance of moving in the intended direction.
- 66.66% chance of moving in orthogonal directions.
This uncertainty adds complexity to the agent's navigation.

### Rewards
- +1 for reaching the goal state (G).
- 0 reward for all other states, including the starting state (S) and intermediate states.

### Episode Termination
The episode terminates when the agent reaches the goal state (G) or falls into a hole (H).

Certainly! Below is the algorithm for the Value Iteration function in Markdown format:

## Value Iteration Algorithm

**Input:**
- Transition probabilities `P`
- Discount factor `gamma`
- Convergence threshold `theta`

**Output:**
- Optimal value function `V`
- Optimal policy `pi`

1. Initialize the value function `V` as an array of zeros, with length equal to the number of states in the problem.

2. **While True:**
   a. Initialize the action-value function `Q` as an array of zeros, with dimensions (number of states, number of actions).
   b. **For each state `s` from 0 to the number of states - 1:**
      i. **For each action `a` from 0 to the number of actions for state `s`:**
         - **For each transition `(prob, next_state, reward, done)` in the transition probabilities for state `s` and action `a`:**
           - Update the action-value function `Q[s][a]` using the Bellman equation:
             ```
             Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
             ```
   c. Calculate the maximum difference between the old `V` and the new `V`:
      ```
      max_diff = np.max(np.abs(V - np.max(Q, axis=1)))
      ```
   d. **If `max_diff` is less than `theta`, break out of the loop. This indicates convergence.**

3. Update the value function `V` with the maximum action-value from `Q`:
   ```
   V = np.max(Q, axis=1)
   ```

4. Compute the optimal policy `pi` based on the action-values in `Q`:
   ```
   pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
   ```

5. **Return:**
   - The optimal value function `V`
   - The optimal policy `pi`

This algorithm iteratively updates the value function `V` until it converges, using the Bellman equation. The optimal policy is then derived from the resulting action-values in `Q`. The algorithm terminates when the maximum difference between old and new `V` is less than the specified convergence threshold `theta`.

## VALUE ITERATION FUNCTION
Below is the Value Iteration function in Python:

```python
def value_iteration(P, gamma=1.0, theta=1e-10):
    # Initialize the value function V as an array of zeros
    V = np.zeros(len(P), dtype=np.float64)
    
    while True:
        # Initialize the action-value function Q as an array of zeros
        Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
        
        for s in range(len(P)):
            for a in range(len(P[s])):
                for prob, next_state, reward, done in P[s][a]:
                    # Update the action-value function Q using the Bellman equation
                    Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
        
        # Check if the maximum difference between Old V and new V is less than theta.
        if np.max(np.abs(V - np.max(Q, axis=1))) < theta:
            break
        
        # Update the value function V with the maximum action-value from Q
        V = np.max(Q, axis=1)

    # Compute the policy pi based on the action-value function Q
    pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
    return V, pi
```
## OUTPUT:
 ![image](https://github.com/Marinto-Richee/Reinforcement-Learning/assets/65499285/dd5cbcb6-a9fe-427f-9eb0-8a469e4c44e4)
 ![image](https://github.com/Marinto-Richee/Reinforcement-Learning/assets/65499285/ba015cf4-6b95-42e9-8422-322ec271c3dc)
 ![image](https://github.com/Marinto-Richee/Reinforcement-Learning/assets/65499285/994f5b78-3c4f-45c4-a398-7b2a7e3f4f2f)



## RESULT:
The result of the experiment will be presented here, including the obtained optimal policy, optimal value function, and success rate for the optimal policy.
