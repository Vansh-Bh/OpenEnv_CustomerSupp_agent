# OpenEnv Customer Support Agent 

An RL environment that simulates real-world e-commerce customer support. An AI agent must triage customer complaints and choose the correct resolution action: issue a refund, send a replacement, reject a fraudulent claim, or ask for proof before proceeding.

---

## Environment Overview

Online retail support teams handle hundreds of tickets daily. Each ticket requires the agent to weigh delivery timelines, proof of damage, and order history to decide the correct action. This environment formalises that decision loop as a sequential RL task, enabling automated evaluation of agent policy quality across increasingly difficult scenarios.

---

## Action Space

The agent must emit exactly one of the following tokens per step:

| Action | When to use |
|--------|-------------|
| `refund` | Defective / damaged item with proof, within policy window |
| `replace` | Wrong or broken item delivered, replacement preferred |
| `reject` | Fraudulent or ineligible claim (used product, tracking mismatch) |
| `ask_proof` | Damage reported but no evidence attached — request images |

---

## Observation Space

Each observation is a plain-text string containing three fields:

```
Customer Query: <free-text complaint>
Days since delivery: <integer>
Proof provided: <True | False>
```

The agent reads this context and returns one action token. After all cases in a task are resolved the environment returns `"All cases completed."` and sets `done=True`.

---

## Tasks

| Task | Cases | Difficulty | Description |
|------|-------|------------|-------------|
| `easy` | 5 | Easy | Genuine complaints with clear evidence. Correct action is unambiguous. |
| `medium` | 5 | Medium | Borderline or policy-violating requests — missing proof, used products, tracking mismatches. |
| `hard` | 5 | Hard | Ambiguous language; agent must recognise uncertainty and ask for proof rather than acting prematurely. |

Tasks cycle automatically: each `/reset` call advances to the next task (easy → medium → hard → easy …).

---

## Reward Function

Rewards are emitted at every step, providing trajectory-level feedback:

| Outcome | Reward |
|---------|--------|
| Correct action | `0.9` |
| `ask_proof` when proof was required | `0.5` |
| Wrong action | `0.1` |

The per-task score is `correct_count / total_cases`, clamped to `(0.05, 0.95)`.

---

## Setup & Usage

### Prerequisites

- Docker
- Python 3.10+
- An OpenAI-compatible API key and endpoint

### Environment Variables

| Variable | Default | Required |
|----------|---------|----------|
| `HF_TOKEN` | — | **Yes** (used as API key) |
| `API_BASE_URL` | `https://api.openai.com/v1` | No |
| `MODEL_NAME` | `gpt-4o-mini` | No |

### Run with Docker

```bash
docker build -t customer-support-env .
docker run -p 7860:7860 customer-support-env
```

### Run inference

```bash
export HF_TOKEN=your_token_here
export API_BASE_URL=https://api.openai.com/v1   # optional
export MODEL_NAME=gpt-4o-mini                   # optional

python inference.py
```

### Local development (without Docker)

```bash
pip install fastapi uvicorn pydantic requests openai
python app.py          # starts the environment server on :7860
python inference.py    # runs the agent against it
```

---

## Expected Output Format

```
[START] task=easy env=openenv model=gpt-4o-mini
[STEP] step=1 action=refund reward=0.90 done=false error=null
[STEP] step=2 action=replace reward=0.90 done=false error=null
...
[END] success=true steps=5 rewards=0.90,0.90,0.90,0.90,0.90

[START] task=medium env=openenv model=gpt-4o-mini
...
[END] success=true steps=5 rewards=0.10,0.90,0.10,0.90,0.90

[START] task=hard env=openenv model=gpt-4o-mini
...
[END] success=true steps=5 rewards=0.50,0.50,0.50,0.50,0.50
```

---

## Baseline Performance (gpt-4o-mini)

| Task | Score |
|------|-------|
| Easy | ~0.90 |
| Medium | ~0.70 |
| Hard | ~0.60 |

Scores are reproducible given the same model and deterministic case ordering.

---

## Project Structure

```
.
├── server # FastAPI server (environment endpoints)
│   ├── app.py           
├── environment.py    # SupportEnv RL loop and grader
├── dataset.py        # Easy / medium / hard case definitions
├── inference.py      # Baseline agent using OpenAI client
├── inference_test.py # Local test script
├── openenv.yaml      # OpenEnv metadata
├── pyproject.toml    # Package definition
└── Dockerfile        # Container build
```