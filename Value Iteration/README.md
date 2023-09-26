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

## POLICY ITERATION ALGORITHM
Include the steps involved in the value iteration algorithm.

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
The output of the Value Iteration algorithm will include:
- Optimal policy
- Optimal value function
- The success rate for the optimal policy
## RESULT:
The result of the experiment will be presented here, including the obtained optimal policy, optimal value function, and success rate for the optimal policy.
