import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from server.app import Env, Action

class CodeReviewEnv(gym.Env):
    """Gym environment wrapper for the AI Code Review RL environment"""
    
    def __init__(self):
        super().__init__()
        self.env = Env()
        
        # Observation space: [step_number, max_steps, num_issues_found, task_easy, task_medium, task_hard]
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0]), 
            high=np.array([100, 100, 10, 1, 1, 1]), 
            dtype=np.float32
        )
        
        # Action space: 0 = no action, 1-20 = flag line 1-20
        self.max_lines = 20
        self.action_space = gym.spaces.Discrete(self.max_lines + 1)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.env.reset()
        return self._obs_to_array(obs), {}
    
    def step(self, action):
        if action == 0:
            line_number = None
        else:
            line_number = action
            
        action_obj = Action(
            action_type="FLAG_BUG",
            line_number=line_number,
            issue_type="unknown",
            comment=""
        )
        
        obs, reward, done, info = self.env.step_env(action_obj)
        return self._obs_to_array(obs), reward, done, False, info
    
    def _obs_to_array(self, obs):
        task_easy = 1.0 if obs.task_name == "easy" else 0.0
        task_medium = 1.0 if obs.task_name == "medium" else 0.0
        task_hard = 1.0 if obs.task_name == "hard" else 0.0
        
        return np.array([
            float(obs.step_number),
            float(obs.max_steps),
            float(len(obs.issues_found_so_far)),
            task_easy,
            task_medium,
            task_hard
        ], dtype=np.float32)

def main():
    print("Creating Code Review RL Environment...")
    env = CodeReviewEnv()
    
    print("Checking environment...")
    check_env(env)
    print("Environment check passed!")
    
    print("Initializing PPO model...")
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0
    )
    
    print("Starting training...")
    model.learn(total_timesteps=5000)  # Reduced for demo
    
    print("Saving model...")
    model.save("ppo_code_review")
    
    print("Evaluating trained model...")
    obs, _ = env.reset()
    total_reward = 0
    steps = 0
    done = False
    
    while not done and steps < 50:
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        steps += 1
        print(f"Step {steps}: Action={action}, Reward={reward:.2f}, Done={done}")
    
    final_score = len(env.env.found) / len(env.env.issues) if env.env.issues else 0
    print(f"Evaluation complete. Total reward: {total_reward:.2f}, Final score: {final_score:.3f}")

if __name__ == "__main__":
    main()