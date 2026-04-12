import os
import requests
import json
from openai import OpenAI

# ✅ STRICT: use ONLY injected environment variables (no defaults)
API_BASE_URL = os.environ["API_BASE_URL"].rstrip("/")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")
API_KEY = os.environ["API_KEY"]

# Environment server URL (same as API_BASE_URL in validation)
ENV_BASE_URL = API_BASE_URL

# ✅ Initialize OpenAI client with THEIR proxy
openai_client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL
)

SYSTEM_PROMPT = """You are an expert code reviewer AI agent. Your task is to analyze code snippets for bugs and issues.

You will receive observations about code to review, including the code snippet, task difficulty, current step, issues found so far, and maximum steps.

Your actions should be in JSON format: {"action_type": "FLAG_BUG", "line_number": <int>, "issue_type": "<type>", "comment": "<explanation>"}

Guidelines:
- Carefully analyze the code for syntax errors, logic bugs, security issues, and design problems
- Flag issues by specifying the line number (1-based) where the issue occurs
- Only flag lines that actually have issues
- Provide accurate issue types and helpful comments
- If you're unsure or want to end the review, you can choose not to flag anything by setting line_number to null or an invalid line
- Be precise and avoid false positives

Respond only with valid JSON action."""


def get_action_from_model(observation):
    """Always use LLM via proxy (no fallback)."""

    code = observation["code_snippet"]
    task = observation["task_name"]
    step = observation["step_number"]
    max_steps = observation["max_steps"]
    found = observation["issues_found_so_far"]

    user_prompt = f"""Task: {task}
Current step: {step}/{max_steps}
Issues found so far: {found}

Code to review:
{code}

What action do you take next? Respond with JSON action."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    try:
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=200,
            temperature=0.1
        )

        generated_text = response.choices[0].message.content or ""

        # Extract JSON safely
        start = generated_text.find("{")
        end = generated_text.rfind("}") + 1

        if start != -1 and end > start:
            try:
                return json.loads(generated_text[start:end])
            except:
                pass

        # Minimal safe fallback (still after API call)
        return {
            "action_type": "FLAG_BUG",
            "line_number": None,
            "issue_type": "none",
            "comment": "Parsing error"
        }
    except Exception as e:
        print(f"[WARN] Error calling proxy API: {e}, using safe fallback", flush=True)
        return {
            "action_type": "FLAG_BUG",
            "line_number": None,
            "issue_type": "none",
            "comment": "API error"
        }


def safe_json_request(method, url, **kwargs):
    try:
        response = requests.request(method, url, timeout=10, **kwargs)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Request failed for {url}: {exc}") from exc

    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError(f"Invalid JSON response from {url}: {exc}") from exc


def validate_action(action):
    if not isinstance(action, dict):
        return False
    if not isinstance(action.get("action_type"), str):
        return False
    line_number = action.get("line_number")
    if line_number is not None and not isinstance(line_number, int):
        return False
    return True


def run_suite():
    """Main execution function - runs all tasks."""
    print(f"[START] task=baseline env=code-review model={MODEL_NAME}", flush=True)

    all_scores = []
    all_rewards = []

    for task_idx in range(3):  # Run through all three tasks
        rewards = []
        steps = 0
        
        try:
            res = safe_json_request("POST", f"{ENV_BASE_URL}/reset")
        except RuntimeError as exc:
            print(f"[ERROR] Failed to reset environment: {exc}", flush=True)
            return
        
        if not isinstance(res, dict) or "task_name" not in res or "max_steps" not in res:
            print(f"[ERROR] Invalid reset response: {res}", flush=True)
            return
            
        done = False
        
        print(f"[TASK_START] task={res['task_name']} max_steps={res['max_steps']}", flush=True)
        
        while not done and steps < 50:  # Safety limit
            action = get_action_from_model(res)
            if not validate_action(action):
                print(f"[WARN] Invalid action from model, using safe default: {action}", flush=True)
                action = {
                    "action_type": "FLAG_BUG",
                    "line_number": None,
                    "issue_type": "none",
                    "comment": "Invalid action"
                }

            try:
                response = safe_json_request("POST", f"{ENV_BASE_URL}/step", json=action)
            except RuntimeError as exc:
                print(f"[ERROR] Failed to send action: {exc}", flush=True)
                return

            if not isinstance(response, dict):
                print(f"[ERROR] Invalid step response: {response}", flush=True)
                return

            try:
                reward = float(response.get("reward", -0.2))
            except (TypeError, ValueError):
                reward = -0.2

            done = bool(response.get("done", False))
            res = response.get("observation", {})
            if not isinstance(res, dict):
                print(f"[ERROR] Invalid observation: {res}", flush=True)
                return
            
            steps += 1
            rewards.append(reward)
            
            print(f"[STEP] step={steps} action={action.get('action_type','unknown')} line={action.get('line_number','none')} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)
            
            if done:
                break
        
        # Calculate final score
        if res and isinstance(res, dict) and "issues_found_so_far" in res:
            final_score = len(res["issues_found_so_far"]) / 4
        else:
            try:
                score_res = safe_json_request("GET", f"{ENV_BASE_URL}/score")
                final_score = float(score_res.get("score", 0))
            except RuntimeError as exc:
                print(f"[WARN] Failed to retrieve score: {exc}", flush=True)
                final_score = 0
        
        all_scores.append(final_score)
        all_rewards.extend(rewards)
        
        task_name = res.get('task_name', 'unknown')
        print(f"[TASK_END] task={task_name} steps={steps} score={final_score:.3f}", flush=True)

    overall_score = sum(all_scores) / len(all_scores) if all_scores else 0
    print(f"[END] success=true total_tasks=3 overall_score={overall_score:.3f}", flush=True)


if __name__ == "__main__":
    try:
        run_suite()
    except requests.exceptions.ConnectionError:
        print("[ERROR] Server offline. Start the server with: python -m uvicorn server.app:app --port 7860", flush=True)
        exit(1)
    except Exception as e:
        print(f"[FATAL] Unhandled exception: {e}", flush=True)
        import traceback
        traceback.print_exc()
        exit(1)