# Policy Iteration Algorithm

## AIM

The aim of this experiment is to find the optimal policy for a given Markov decision process (MDP) using the policy iteration algorithm.

## PROBLEM STATEMENT

The problem statement is as follows:

* Given an MDP with a set of states, actions, and transition probabilities, we want to find the optimal policy.
* The optimal policy is the policy that maximizes the expected discounted reward over the long term.

## POLICY ITERATION ALGORITHM

The policy iteration algorithm is a dynamic programming algorithm that solves the problem of finding the optimal policy for an MDP. It works by iteratively improving a policy until it converges to the optimal policy.

The following are the steps involved in the policy iteration algorithm:

1. Initialize a policy.
2. Evaluate the policy using the policy evaluation algorithm.
3. Improve the policy using the policy improvement algorithm.
4. Repeat steps 2 and 3 until the policy converges to the optimal policy.

## POLICY IMPROVEMENT FUNCTION

The policy improvement function takes a value function and a transition probability matrix as input and returns a new policy. The new policy is the policy that maximizes the expected discounted reward for each state, given the value function.

The following is a simple implementation of the policy improvement function:

```python
def policy_improvement(V, P, gamma=1.0):
    """
    Computes the policy that maximizes the expected discounted reward for each state, given the value function.

    Args:
        V: The value function.
        P: The transition probability matrix.
        gamma: The discount factor.

    Returns:
        The policy that maximizes the expected discounted reward for each state.
    """

    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    for s in range(len(P)):
        for a in range(len(P[s])):
            for prob, next_state, reward, done in P[s][a]:
                Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
    new_pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
    return new_pi
```
## POLICY ITERATION FUNCTION
The following is a simple implementation of the policy iteration function:
```python3
def policy_iteration(P, gamma=1.0, theta=1e-10):
    """
    Finds the optimal policy for a given MDP using the policy iteration algorithm.

    Args:
        P: The transition probability matrix.
        gamma: The discount factor.
        theta: The error tolerance.

    Returns:
        The optimal policy and the optimal value function.
    """

    random_actions = np.random.choice(tuple(P[0].keys()), len(P))
    pi = lambda s: {s: a for s, a in enumerate(random_actions)}[s]
    while True:
        old_pi = {s: pi(s) for s in range(len(P))}
        V = policy_evaluation(pi, P, gamma, theta)
        pi = policy_improvement(V, P, gamma)
        if old_pi == {s: pi(s) for s in range(len(P))}:
            break
    return V, pi
```

## Output

| Gamma | State | Optimal Action | State Value |
|---|---|---|---|
| 1.0 | 01 | 1 | 0.09537 |
| 1.0 | 02 | 0 | 0.17485 |
| 1.0 | 03 | 0 | 0.28877 |
| 1.0 | 04 | 0 | 0.47113 |
| 1.0 | 05 | 0 | 0.76748 |
| 0.8 | 01 | 1 | 0.0763 |
| 0.8 | 02 | 0 | 0.12212 |
| 0.8 | 03 | 0 | 0.19562 |
| 0.8 | 04 | 0 | 0.30325 |
| 0.8 | 05 | 0 | 0.48488 |

## Conclusion

The policy iteration experiment shows that the policy iteration algorithm is a powerful tool for finding the optimal policy for an MDP. The optimal policy is the policy that maximizes the expected discounted reward over the long term, regardless of the value of the discount factor gamma.

The optimal policy for both values of gamma is to take action 1 in state 01 and action 0 in all other states. However, the state values are higher for all states when gamma is 1.0. This is because a higher value of gamma places more weight on future rewards.
