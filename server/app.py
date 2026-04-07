from fastapi import FastAPI
from pydantic import BaseModel
from environment import SupportEnv
from dataset import easy_cases, medium_cases, hard_cases
import random

app = FastAPI()

# Models 
class Action(BaseModel):
    message: str

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

# Environment (dynamic tasks) 
def create_env():
    selected_cases = random.choice([
        easy_cases,
        medium_cases,
        hard_cases
    ])
    return SupportEnv(selected_cases)

env = create_env()

# Routes 
@app.post("/reset")
def reset():
    global env
    env = create_env()   # NEW TASK EACH TIME

    obs = env.reset()

    return {
        "observation": {
            "echoed_message": obs
        },
        "reward": 0.0,
        "done": False
    }

@app.post("/step")
def step(action: Action):
    parsed_action = parse_action(action.message)

    obs, reward, done = env.step(parsed_action)

    return {
        "observation": {
            "echoed_message": obs
        },
        "reward": reward,
        "done": done
    }

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
