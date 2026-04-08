import random

class SupportEnv:
    def __init__(self, cases):
        self.cases = cases
        self.index = 0
        self.correct_count = 0

    def reset(self):
        self.index = 0
        self.correct_count = 0
        return self._get_obs()

    def step(self, action):
        case = self.cases[self.index]
        correct = case["correct_action"]

        if action == correct:
            reward = 0.9
            self.correct_count += 1
        elif action == "ask_proof" and correct == "ask_proof":
            reward = 0.5
        else:
            reward = 0.1

        self.index += 1
        done = self.index >= len(self.cases)

        obs = self._get_obs() if not done else "All cases completed."

        return obs, reward, done

    # TASK GRADER
    def get_score(self):
        total = len(self.cases)
        score = self.correct_count / total if total > 0 else 0.0

        # ensure strict range
        return max(0.01, min(0.99, score))

    def _get_obs(self):
        case = self.cases[self.index]

        return f"""
Customer Query: {case['query']}
Days since delivery: {case['days']}
Proof provided: {case.get('has_proof', False)}

Choose one action:
refund / replace / reject / ask_proof
"""
