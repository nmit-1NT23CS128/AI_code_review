# AI Code Review Environment

## Environment Overview and Motivation

This is a realistic reinforcement learning environment built for the OpenEnv Hackathon. An AI agent acts as a code reviewer: it receives code snippets containing real bugs and must flag, explain, and fix issues — earning rewards for accuracy and losing points for false alarms.

Every engineering team does code review. It's a high-value, well-defined task with:
- Clear correct/incorrect states (expected issues are ground-truth)
- Progressive difficulty (syntax → logic → architecture)
- Deterministic grading (no human judgment needed)
- Rich reward signal (partial credit, severity weighting)

This environment teaches an AI agent to review code the way a senior engineer would: systematically, precisely, and with clear explanations.

## Action and Observation Spaces

### Observation
```python
class Observation(BaseModel):
    code_snippet: str          # The code to review
    step_number: int           # Current step in the episode
    max_steps: int             # Maximum steps allowed
    issues_found_so_far: List[int]  # Line numbers of issues already flagged
    task_name: str             # Difficulty level: "easy", "medium", or "hard"
```

### Action
```python
class Action(BaseModel):
    action_type: str           # Currently "FLAG_BUG"
    line_number: Optional[int] # Line number to flag (1-based)
    issue_type: Optional[str]  # Type of issue (e.g., "syntax", "logic", "security")
    comment: Optional[str]     # Explanation of the issue
```

### Reward
- +1.0: Correctly flagging a new issue
- -0.2: Incorrectly flagging a line or taking invalid action
- Rewards are given per step, encouraging incremental progress

## Tasks

### Task 1 — Easy: fix_syntax_and_obvious_bugs
**Code**: A short Python function to calculate an average  
**Issues (3)**:
- Off-by-one error in range() → IndexError
- Syntax error: = instead of == in if condition
- Potential ZeroDivisionError for empty input  
**Max steps**: 8  
**Grader**: issues_correctly_found / 3

### Task 2 — Medium: logic_and_security_review
**Code**: A Flask-style route handler querying a database  
**Issues (4)**:
- SQL Injection via f-string interpolation (critical)
- No authentication check before returning user data
- Implicit None return instead of 404 response
- Database connection never closed (resource leak)  
**Max steps**: 12  
**Grader**: issues_correctly_found / 4

### Task 3 — Hard: design_and_architecture_review
**Code**: An OrderProcessor class managing a pipeline  
**Issues (5)**:
- Hardcoded database password in source code
- Mutable default argument (pending_orders=[])
- SRP violation: email sending inside order processor
- Swallowed exception with bare except: pass
- Thread-safety bug: read-modify-write without lock  
**Max steps**: 18  
**Grader**: issues_correctly_found / 5

## Setup and Usage

### Local Development
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the environment server:
   ```bash
   python -m server.app
   ```

3. In another terminal, run the baseline inference:
   ```bash
   export HF_TOKEN="your-huggingface-token"
   python inference.py
   ```

### Docker
```bash
docker build -t ai-code-review .
docker run -p 7860:7860 ai-code-review
```

### OpenEnv Validation
```bash
pip install openenv-core
openenv validate
```

## Baseline Performance Scores

Using Google Flan-T5-Base via Hugging Face API:
- Easy: 0.333 (1/3 issues found)
- Medium: 0.250 (1/4 issues found)
- Hard: 0.200 (1/5 issues found)
- Overall: 0.261

*Note: Baseline scores may vary based on model and prompt engineering. The environment cycles through tasks, so running the script multiple times will evaluate all tasks.*

## Alternative Models

You can use different Hugging Face models by setting:
```bash
export MODEL_NAME="microsoft/codebert-base"  # or other HF models
```

## Deployment

This environment is designed for deployment on Hugging Face Spaces. Tag your space with `openenv` for discoverability.

The containerized application exposes a FastAPI server on port 7860 with endpoints for reset, step, state, and score.