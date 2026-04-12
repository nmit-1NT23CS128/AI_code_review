import os
import requests
import json

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860")
MODEL_NAME = os.getenv("MODEL_NAME", "google/flan-t5-base")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required")

HF_API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"

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

    full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
    
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": full_prompt,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.1,
            "do_sample": True
        }
    }
    
    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
        if isinstance(result, list) and result:
            generated_text = result[0].get("generated_text", "")
        else:
            generated_text = str(result)
        
        # Extract JSON from the response
        # Look for JSON in the generated text
        start = generated_text.find("{")
        end = generated_text.rfind("}") + 1
        if start != -1 and end > start:
            json_str = generated_text[start:end]
            action = json.loads(json_str)
            return action
        else:
            # Fallback
            return {
                "action_type": "FLAG_BUG",
                "line_number": None,
                "issue_type": "none",
                "comment": "Model error"
            }
    except Exception as e:
        print(f"Error getting action from model: {e}")
        # Fallback to no action
        return {
            "action_type": "FLAG_BUG",
            "line_number": None,
            "issue_type": "none",
            "comment": "Model error"
        }


print(f"[START] task=baseline env=code-review model={MODEL_NAME}")

all_scores = []
all_rewards = []

for task_idx in range(3):  # Run through all three tasks
    rewards = []
    steps = 0
    
    try:
        res = requests.post(f"{API_BASE_URL}/reset").json()
    except Exception as e:
        print(f"[ERROR] Failed to reset environment: {e}")
        continue
    
    if "task_name" not in res or "max_steps" not in res:
        print(f"[ERROR] Invalid reset response: {res}")
        continue
        
    done = False
    
    print(f"[TASK_START] task={res['task_name']} max_steps={res['max_steps']}")
    
    while not done and steps < 50:  # Safety limit
        action = get_action_from_model(res)
        
        try:
            response = requests.post(f"{API_BASE_URL}/step", json=action).json()
        except Exception as e:
            print(f"[ERROR] Failed to step environment: {e}")
            break
        
        reward = float(response.get("reward", -0.2))
        done = response.get("done", False)
        res = response.get("observation", {})
        
        steps += 1
        rewards.append(reward)
        
        print(f"[STEP] step={steps} action={action.get('action_type','unknown')} line={action.get('line_number','none')} reward={reward:.2f} done={str(done).lower()} error=null")
        
        if done:
            break
    
    score = sum(rewards) / len(rewards) if rewards else 0
    score = max(0, min(score, 1))
    
    if res and "issues_found_so_far" in res:
        final_score = len(res["issues_found_so_far"]) / 4  # Approximate max issues
    else:
        try:
            score_res = requests.get(f"{API_BASE_URL}/score").json()
            final_score = score_res.get("score", 0)
        except:
            final_score = 0
    
    all_scores.append(final_score)
    all_rewards.extend(rewards)
    
    task_name = res.get('task_name', 'unknown')
    print(f"[TASK_END] task={task_name} steps={steps} score={final_score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}")

overall_score = sum(all_scores) / len(all_scores) if all_scores else 0
print(f"[END] success=true total_tasks=3 overall_score={overall_score:.3f} all_scores={','.join(f'{s:.3f}' for s in all_scores)}")