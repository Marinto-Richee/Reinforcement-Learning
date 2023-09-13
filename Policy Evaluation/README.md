# Policy Evaluation

## AIM

The aim of this experiment is to evaluate the performance of two policies using policy evaluation.

## PROBLEM STATEMENT

The problem statement is as follows:

* Given a set of states, actions, and transition probabilities, we are given two policies.
* We want to evaluate the performance of the two policies by computing their state-value functions.
* The policy with the higher state-value function is considered to be the better policy.

## POLICY EVALUATION FUNCTION

The policy evaluation function is as follows:

```python
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    """
    Evaluates the given policy using policy iteration.

    Args:
        pi: The policy to evaluate.
        P: The transition probabilities.
        gamma: The discount factor.
        theta: The error tolerance.

    Returns:
        The value function of the given policy.
    """
    V = np.zeros(len(P), dtype=np.float64)
    i = 0
    while True:
      V=np.zeros(len(P),dtype=np.float64)
      for s in range(len(P)):
        for prob, next_state, reward, done in P[s][pi(s)]:
          V[s]+=prob*(reward+gamma*V[next_state]*(not done))
      if np.max(np.abs(prev_V-V))<theta:
        break
      prev_V=V.copy()
    return V

```

## OUTPUT

The output of the policy evaluation experiment is as follows:

* The state-value functions of the two policies.
* A comparison of the two policies based on their state-value functions.
  
Here is a table that summarizes the state-value functions of the two policies:

State | Policy 1 | Policy 2
------- | -------- | --------
01 | 2.9533 | 5.67308
02 | 5.81319 | 16.69231
03 | 8.39286 | 18.36538
04 | 10.13187 | 16.92308
05 | 9.3489 | 14.44231

## RESULT
The result of the policy evaluation experiment is as follows:
* Policy 2 has a higher state-value function than Policy 1 for all states.
* Therefore, Policy 2 is the better policy.
  
The state with the highest difference in value between the two policies is state 01, where Policy 2 has a value that is 2.71991 higher than Policy 1. This means that Policy 2 is expected to have a significantly higher return in state 01 than Policy 1.
