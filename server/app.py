from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

class Observation(BaseModel):
    code_snippet: str
    step_number: int
    max_steps: int
    issues_found_so_far: List[int]
    task_name: str

class Action(BaseModel):
    action_type: str
    line_number: Optional[int] = None
    issue_type: Optional[str] = None
    comment: Optional[str] = None

class Env:
    def __init__(self):
        self.tasks = {
            "easy": {
                "code": """def calculate_average(nums):
    if len(nums) = 0:
        return 0
    total = 0
    for i in range(len(nums) + 1):
        total += nums[i]
    return total / len(nums)""",
                "issues": [2, 5, 8],
                "max_steps": 8
            },
            "medium": {
                "code": """@app.route('/user/<user_id>')
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    result = db.execute(query).fetchone()
    return result""",
                "issues": [3, 4, 5, 6],
                "max_steps": 12
            },
            "hard": {
                "code": """class OrderProcessor:
    def __init__(self):
        self.db_password = "secret123"
    def process_orders(self, pending_orders=[]):
        for order in pending_orders:
            try:
                self.save_to_db(order)
                self.send_email(order.customer_email, "Order processed")
            except:
                pass
        pending_orders.append(order)""",
                "issues": [3, 4, 6, 7, 8],
                "max_steps": 18
            }
        }
        self.task_names = list(self.tasks.keys())
        self.current_task_index = 0
        self.reset()

    def reset(self):
        # Cycle through tasks
        self.task_name = self.task_names[self.current_task_index]
        self.current_task_index = (self.current_task_index + 1) % len(self.task_names)
        
        task = self.tasks[self.task_name]
        self.code = task["code"]
        self.issues = task["issues"]
        self.max_steps = task["max_steps"]
        self.found = []
        self.step = 0
        return self.state()

    def step_env(self, action: Action):
        reward = 0.0
        done = False

        if action.line_number is not None and action.line_number in self.issues and action.line_number not in self.found:
            reward = 1.0
            self.found.append(action.line_number)
        else:
            reward = -0.2

        self.step += 1
        if self.step >= self.max_steps:
            done = True

        return self.state(), reward, done, {}

    def state(self):
        return Observation(
            code_snippet=self.code,
            step_number=self.step,
            max_steps=self.max_steps,
            issues_found_so_far=self.found,
            task_name=self.task_name
        )

env = Env()

@app.post("/reset")
def reset():
    return env.reset()

@app.post("/step")
def step(action: Action):
    obs, reward, done, info = env.step_env(action)
    return {"observation": obs, "reward": reward, "done": done, "info": info}

@app.get("/state")
def state():
    return env.state()

@app.get("/score")
def score():
    return {"score": len(env.found) / len(env.issues)}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"status": "ok"}


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()