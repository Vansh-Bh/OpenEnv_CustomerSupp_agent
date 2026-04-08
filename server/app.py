from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from dataset import easy_cases, hard_cases, medium_cases
from environment import SupportEnv

app = FastAPI()

# --- Models ---
class Action(BaseModel):
    message: str

class ResetRequest(BaseModel):
    task: str = "easy"

class Observation(BaseModel):
    echoed_message: str

class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict = {}

# --- Parser ---
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

# --- TASKS ---
TASKS = {"easy": easy_cases, "medium": medium_cases, "hard": hard_cases}

env = None

# --- Routes ---

# RESET
@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    global env
    
    task_name = req.task if req and req.task in TASKS else "easy"

    cases = TASKS[task_name]
    env = SupportEnv(cases)

    obs = env.reset()

    return {
        "observation": {"echoed_message": obs}, 
        "reward": 0.0, 
        "done": False, 
        "info": {}
    }

# STATE
@app.get("/state")
def get_state():
    global env
    if env is None:
        return {"error": "Environment not initialized. Call /reset first."}

    return {"observation": {"echoed_message": env._get_obs()}}

# STEP
@app.post("/step", response_model=StepResponse)
def step(action: Action):
    global env

    # Safety check if step is called before reset
    if env is None:
        return {
            "observation": {"echoed_message": "Error: Environment not initialized."},
            "reward": 0.0,
            "done": True,
            "info": {}
        }

    parsed_action = parse_action(action.message)

    obs, reward, done = env.step(parsed_action)

    response = {
        "observation": {"echoed_message": obs}, 
        "reward": reward, 
        "done": done, 
        "info": {}
    }

    if done:
        response["info"]["score"] = float(env.get_score())

    return response

# --- Main ---
def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
