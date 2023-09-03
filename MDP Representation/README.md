# Windmill Agent MDP Representation

## AIM:
The aim of this MDP representation is to model the decision-making process for a windmill agent operating in a wind farm to optimize its energy generation.

## PROBLEM STATEMENT:
### Problem Description
The problem is to determine the optimal actions for the windmill agent to control the orientation and operation of the windmill blades in order to maximize energy production while taking into account the uncertain and stochastic nature of wind conditions.

## State Space:
The state space for the windmill agent includes various environmental and internal factors, such as:
- Wind speed and direction.
- Current energy production level.
- Health status of the windmill components (e.g., blade wear, gearbox condition).
- Time of day.

## Sample State:
Here's an example of a state in the state space:
- Wind speed: 10 m/s
- Wind direction: 220 degrees
- Energy production: 500 kWh
- Blade wear: 20%
- Time of day: 12:00 PM

## Action Space:
The action space represents the set of actions that the windmill agent can take, such as:
- Adjust the blade pitch angle.
- Changing the rotor speed.
- Performing maintenance if necessary.

## Sample Action:
Here's an example of an action in the action space:
- Increase blade pitch angle by 5 degrees.

## Reward Function:
The reward function assigns a numerical value to the agent's actions in each state, indicating how good or bad the action is with respect to achieving the goal. In this MDP, the reward function could be designed to maximize energy production while minimizing wear and maintenance costs. For example:
- Positive reward for increasing energy production.
- Negative reward for excessive wear or maintenance actions.

## Graphical Representation:
![diagram](https://github.com/Marinto-Richee/Reinforcement-Learning/assets/65499285/0e504d6c-9be0-49ce-941e-5280a5809af5)

## PYTHON REPRESENTATION:
```python3
windmill_mdp = {
    "LowWind-LowProduction": {
        "AdjustBladePitch": [(0.8, "LowWind-HighProduction", 5.0, False), (0.2, "LowWind-LowProduction", 0.0, False)],
        "ChangeRotorSpeed": [(0.7, "LowWind-ModerateProduction", 2.0, False), (0.3, "LowWind-LowProduction", 1.0, False)],
    },
    "LowWind-ModerateProduction": {
        "AdjustBladePitch": [(0.6, "LowWind-HighProduction", 3.0, False), (0.4, "LowWind-ModerateProduction", 0.0, False)],
        "ChangeRotorSpeed": [(0.5, "LowWind-ModerateProduction", 1.0, False), (0.5, "LowWind-LowProduction", 0.0, False)],
    },
    "ModerateWind-HighProduction": {
        "AdjustBladePitch": [(0.9, "ModerateWind-HighProduction", 10.0, False), (0.1, "ModerateWind-ModerateProduction", 5.0, False)],
        "ChangeRotorSpeed": [(0.8, "ModerateWind-ModerateProduction", 2.0, False), (0.2, "ModerateWind-HighProduction", 1.0, False)],
    },
    "HighWind-ModerateProduction": {
        "AdjustBladePitch": [(0.7, "HighWind-HighProduction", 5.0, False), (0.3, "HighWind-ModerateProduction", 1.0, False)],
        "ChangeRotorSpeed": [(0.6, "HighWind-HighProduction", 3.0, False), (0.4, "HighWind-ModerateProduction", 0.0, False)],
    },
    "HighWind-HighProduction": {
        "AdjustBladePitch": [(0.8, "HighWind-HighProduction", 10.0, False), (0.2, "HighWind-ModerateProduction", 5.0, False)],
        "ChangeRotorSpeed": [(1.0, "HighWind-ModerateProduction", 0.0, True)],
    }
}
```
In this Python representation:<br>
- Each state is represented as a dictionary key.
- Actions ("AdjustBladePitch" and "ChangeRotorSpeed") are defined for each state.
- Each action has a list of possible outcomes, where each outcome is represented as a tuple with the format (probability, next_state, reward, is_terminal).

## Result:
Hence, we have successfully represented MDP for the Windmill Agent along with its python representation.
