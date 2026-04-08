from fastapi import FastAPI
from pydantic import BaseModel
from environment import SupportEnv
from dataset import easy_cases, medium_cases, hard_cases

app = FastAPI()

#  Models
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

# TASKS 
TASKS = {
    "easy": easy_cases,
    "medium": medium_cases,
    "hard": hard_cases
}

env = None

# RESET 
@app.post("/reset")
def reset(req: ResetRequest):
    global env

    cases = TASKS.get(req.task, easy_cases)
    env = SupportEnv(cases)

    obs = env.reset()

    return {
        "observation": {"echoed_message": obs},
        "reward": 0.0,
        "done": False
    }

#STEP
@app.post("/step")
def step(action: Action):
    parsed_action = parse_action(action.message)

    obs, reward, done = env.step(parsed_action)

    response = {
        "observation": {"echoed_message": obs},
        "reward": reward,
        "done": done
    }

    #  RETURN TASK SCORE
    if done:
        response["score"] = env.get_score()

    return response

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
