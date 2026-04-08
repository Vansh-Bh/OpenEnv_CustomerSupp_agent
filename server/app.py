from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from environment import SupportEnv
from dataset import easy_cases, medium_cases, hard_cases

app = FastAPI()

# Models 
class Action(BaseModel):
    message: str

class ResetRequest(BaseModel):
    task: str = "easy"

# Parser 
def parse_action(message):
    message = message.lower()

    if "refund" in message:
        return "refund"
    elif "replace" in message:
        return "replace"
    elif "reject" in message:
        return "reject"
    elif "proof" in message:
        return "ask_proof"
    else:
        return "reject"

#  TASKS 
TASKS = {
    "easy": easy_cases,
    "medium": medium_cases,
    "hard": hard_cases
}

# AUTO TASK CYCLING
TASK_SEQUENCE = ["easy", "medium", "hard"]
current_task_index = 0

env = None

# RESET 
@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    global env, current_task_index

    # IGNORE BODY → AUTO CYCLE TASKS
    task = TASK_SEQUENCE[current_task_index]
    current_task_index = (current_task_index + 1) % 3

    cases = TASKS[task]
    env = SupportEnv(cases)

    obs = env.reset()

    return {
        "observation": {"echoed_message": obs},
        "reward": 0.0,
        "done": False
    }

#  STEP
@app.post("/step")
def step(action: Action):
    parsed_action = parse_action(action.message)

    obs, reward, done = env.step(parsed_action)

    response = {
        "observation": {"echoed_message": obs},
        "reward": reward,
        "done": done
    }

    #  RETURN SCORE PER TASK
    if done:
        response["score"] = env.get_score()

    return response

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
