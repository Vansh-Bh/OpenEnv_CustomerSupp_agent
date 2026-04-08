import asyncio
import os
import requests
import time

#  REQUIRED VARIABLES 
API_BASE_URL = os.getenv("API_BASE_URL", "dummy")
MODEL_NAME = os.getenv("MODEL_NAME", "dummy")
HF_TOKEN = os.getenv("HF_TOKEN")

# CONFIG 
SPACE_URL = "https://arrowman123-customer-supp-env.hf.space"

TASK_NAME = "customer-support"
BENCHMARK = "openenv"

MAX_STEPS = 20
MAX_TOTAL_REWARD = 20
SUCCESS_SCORE_THRESHOLD = 0.6


# DUMMY AGENT
def get_model_message(state):
    state = state.lower()

    if "damaged" in state or "broken" in state:
        return "refund"
    elif "wrong item" in state:
        return "replace"
    elif "not sure" in state or "seems" in state:
        return "ask_proof"
    else:
        return "reject"


#  SAFE REQUEST 
def safe_post(url, json=None, retries=3):
    for i in range(retries):
        try:
            return requests.post(url, json=json, timeout=10)
        except Exception as e:
            print(f"[DEBUG] retry {i+1} due to {e}")
            time.sleep(2)
    raise Exception("Failed after retries")


# LOGGING 
def log_start(**kwargs):
    print("[START]", kwargs, flush=True)


def log_step(**kwargs):
    print("[STEP]", kwargs, flush=True)


def log_end(**kwargs):
    print("[END]", kwargs, flush=True)


#  MAIN 
async def main():
    rewards = []
    steps_taken = 0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        # RESET
        res = safe_post(f"{SPACE_URL}/reset")
        result = res.json()

        state = result["observation"]["echoed_message"]
        done = result["done"]

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            message = get_model_message(state)

            res = safe_post(
                f"{SPACE_URL}/step",
                json={"message": message}
            )
            result = res.json()

            reward = result.get("reward", 0.0)
            done = result.get("done", False)
            state = result["observation"]["echoed_message"]

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=message,
                reward=reward,
                done=done,
                error=None
            )

        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = max(0.0, min(1.0, score))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[ERROR] {e}")

    finally:
        log_end(
            success=success,
            steps=steps_taken,
            score=score if 'score' in locals() else 0.0,
            rewards=rewards
        )


#  ENTRY 
if __name__ == "__main__":
    asyncio.run(main())
