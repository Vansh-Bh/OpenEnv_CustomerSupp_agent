import random

class SupportEnv:
    def __init__(self, cases):
        self.cases = cases
        self.index = 0

    def reset(self):
        self.index = 0
        return self._get_obs()

    def step(self, action):
        case = self.cases[self.index]
        correct = case["correct_action"]

        # reward logic
        if action == correct:
            reward = 0.9
        elif action == "ask_proof" and correct == "ask_proof":
            reward = 0.5
        else:
            reward = 0.1

        # move to next case
        self.index += 1
        done = self.index >= len(self.cases)

        obs = self._get_obs() if not done else "All cases completed."

        return obs, reward, done

    def _get_obs(self):
        case = self.cases[self.index]

        return f"""
Customer Query: {case['query']}
Days since delivery: {case['days']}
Proof provided: {case.get('has_proof', False)}

Choose one action:
refund / replace / reject / ask_proof
"""
