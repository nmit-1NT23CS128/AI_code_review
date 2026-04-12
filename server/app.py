from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

app = FastAPI()

class Observation(BaseModel):
    code_snippet: str
    step_number: int
    max_steps: int
    issues_found_so_far: List[int]
    task_name: str
    score: float

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
                "issues": [2, 5, 7],
                "max_steps": 8
            },
            "medium": {
                "code": """@app.route('/user/<user_id>')
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    result = db.execute(query).fetchone()
    return result""",
                "issues": [1, 3, 4, 5],
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
            task_name=self.task_name,
            score=len(self.found) / len(self.issues) if self.issues else 0.0
        )

env = Env()

@app.post("/reset", response_model=Observation)
def reset():
    return env.reset()

@app.post("/step")
def step(action: Action) -> Dict[str, Any]:
    obs, reward, done, info = env.step_env(action)
    return {
        "observation": obs.model_dump() if hasattr(obs, 'model_dump') else obs.dict(),
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/state")
def state():
    return env.state()

@app.get("/score")
def score():
    return {"score": len(env.found) / len(env.issues)}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Code Review Environment</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #1e1e1e; color: #e0e0e0; }
            .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
            header { text-align: center; margin-bottom: 30px; border-bottom: 2px solid #0066cc; padding-bottom: 20px; }
            h1 { color: #00bfff; font-size: 2.5em; margin-bottom: 10px; }
            .subtitle { color: #888; font-size: 1.1em; }
            
            .main-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
            .panel { background: #2d2d2d; border: 1px solid #0066cc; border-radius: 8px; padding: 20px; }
            .panel h2 { color: #00bfff; margin-bottom: 15px; font-size: 1.3em; }
            
            .code-display { background: #1a1a1a; border: 1px solid #444; padding: 15px; border-radius: 5px; font-family: monospace; line-height: 1.6; overflow-x: auto; max-height: 400px; }
            .code-display pre { margin: 0; }
            .code-line { padding: 2px 5px; }
            .code-line.issue { background: #6b2c2c; color: #ff6b6b; }
            .code-line.found { background: #2c6b2c; color: #6bff6b; }
            
            .status { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 15px 0; }
            .stat { background: #1a1a1a; padding: 10px; border-radius: 5px; border-left: 3px solid #0066cc; }
            .stat-label { color: #888; font-size: 0.9em; }
            .stat-value { color: #00bfff; font-size: 1.8em; font-weight: bold; }
            
            .action-panel { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin: 15px 0; }
            button { background: #0066cc; color: white; border: none; padding: 10px 15px; border-radius: 5px; cursor: pointer; font-size: 1em; }
            button:hover { background: #0052a3; }
            button:disabled { background: #555; cursor: not-allowed; }
            input { background: #1a1a1a; border: 1px solid #0066cc; color: #e0e0e0; padding: 8px; border-radius: 3px; }
            
            .reward-display { font-size: 1.2em; padding: 10px; border-radius: 5px; text-align: center; margin: 10px 0; }
            .reward-positive { background: #2c6b2c; color: #6bff6b; }
            .reward-negative { background: #6b2c2c; color: #ff6b6b; }
            
            .full-width { grid-column: 1 / -1; }
            .message { padding: 10px; border-radius: 5px; background: #2c3e50; border-left: 3px solid #0066cc; }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>🔍 AI Code Review Environment</h1>
                <p class="subtitle">Reinforcement Learning for Bug Detection</p>
            </header>
            
            <div class="main-grid">
                <div class="panel">
                    <h2>Code to Review</h2>
                    <div class="code-display">
                        <pre id="codeDisplay">Loading...</pre>
                    </div>
                </div>
                
                <div class="panel">
                    <h2>Episode Status</h2>
                    <div class="status">
                        <div class="stat">
                            <div class="stat-label">Task</div>
                            <div class="stat-value" id="taskName">-</div>
                        </div>
                        <div class="stat">
                            <div class="stat-label">Step</div>
                            <div class="stat-value" id="stepCount">-/-</div>
                        </div>
                        <div class="stat">
                            <div class="stat-label">Issues Found</div>
                            <div class="stat-value" id="issuesFound">0</div>
                        </div>
                        <div class="stat">
                            <div class="stat-label">Score</div>
                            <div class="stat-value" id="score">0%</div>
                        </div>
                    </div>
                    <div id="rewardDisplay" style="display:none;"></div>
                    <button onclick="resetEnv()" style="width: 100%; margin-top: 10px;">🔄 Reset Episode</button>
                </div>
            </div>
            
            <div class="panel full-width">
                <h2>Flag a Bug</h2>
                <div class="action-panel">
                    <input type="number" id="lineNumber" placeholder="Line number" min="1" max="20">
                    <input type="text" id="issueType" placeholder="Issue type (e.g., syntax, logic, security)">
                    <input type="text" id="comment" placeholder="Brief explanation">
                </div>
                <button onclick="flagBug()" style="width: 100%;">✅ Submit Action</button>
                <div id="message" class="message" style="display:none; margin-top: 10px;"></div>
            </div>
        </div>
        
        <script>
            async function resetEnv() {
                try {
                    const response = await fetch('/reset', { method: 'POST' });
                    const data = await response.json();
                    updateUI(data);
                    showMessage('✅ Episode reset!', 'success');
                } catch (e) {
                    showMessage('Error resetting: ' + e.message, 'error');
                }
            }
            
            async function flagBug() {
                const lineNumber = parseInt(document.getElementById('lineNumber').value);
                const issueType = document.getElementById('issueType').value;
                const comment = document.getElementById('comment').value;
                
                if (!lineNumber || !issueType || !comment) {
                    showMessage('Please fill all fields', 'error');
                    return;
                }
                
                try {
                    const response = await fetch('/step', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            action_type: 'FLAG_BUG',
                            line_number: lineNumber,
                            issue_type: issueType,
                            comment: comment
                        })
                    });
                    
                    const data = await response.json();
                    const reward = data.reward;
                    const done = data.done;
                    
                    updateUI(data.observation);
                    showReward(reward);
                    
                    document.getElementById('lineNumber').value = '';
                    document.getElementById('issueType').value = '';
                    document.getElementById('comment').value = '';
                    
                    if (done) {
                        showMessage('✅ Episode finished! Click Reset to start a new one.', 'success');
                    }
                } catch (e) {
                    showMessage('Error: ' + e.message, 'error');
                }
            }
            
            function updateUI(obs) {
                document.getElementById('taskName').textContent = obs.task_name.toUpperCase();
                document.getElementById('stepCount').textContent = obs.step_number + '/' + obs.max_steps;
                document.getElementById('issuesFound').textContent = obs.issues_found_so_far.length;
                
                const codeLines = obs.code_snippet.split('\\n');
                let htmlCode = '';
                codeLines.forEach((line, idx) => {
                    const lineNum = idx + 1;
                    htmlCode += '<span class="code-line' + (obs.issues_found_so_far.includes(lineNum) ? ' found' : '') + '">' + lineNum + ': ' + (line || ' ') + '</span>\\n';
                });
                document.getElementById('codeDisplay').innerHTML = '<pre>' + htmlCode + '</pre>';
                
                const scorePercent = Math.round((obs.score || 0) * 100);
                document.getElementById('score').textContent = scorePercent + '%';
            }
            
            async function updateScore() {
                try {
                    const response = await fetch('/score');
                    const data = await response.json();
                    const scorePercent = Math.round(data.score * 100);
                    document.getElementById('score').textContent = scorePercent + '%';
                } catch (e) {}
            }
            
            function showReward(reward) {
                const display = document.getElementById('rewardDisplay');
                const rewardText = reward > 0 ? '✅ +' + reward.toFixed(2) : '❌ ' + reward.toFixed(2);
                const className = reward > 0 ? 'reward-positive' : 'reward-negative';
                display.innerHTML = '<div class="reward-display ' + className + '">' + rewardText + '</div>';
                display.style.display = 'block';
                setTimeout(() => { display.style.display = 'none'; }, 3000);
            }
            
            function showMessage(msg, type) {
                const msgEl = document.getElementById('message');
                msgEl.textContent = msg;
                msgEl.style.display = 'block';
                msgEl.style.borderLeftColor = type === 'error' ? '#ff6b6b' : '#6bff6b';
            }
            
            // Load initial state
            window.onload = () => { resetEnv(); };
        </script>
    </body>
    </html>
    """


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()