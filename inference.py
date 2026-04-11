import os
import json
from openai import OpenAI

# ==============================================================================
# 1. Environment Variables & Setup
# ==============================================================================
# Read environment variables with defaults where required
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# Initialize OpenAI client as mandated by the rules
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

# ==============================================================================
# 2. Mock Customer Support Environment
# ==============================================================================
# Since this is an agent interaction loop, we set up a simulated environment
# for receiving customer messages (observations) and sending replies (actions).
class CustomerSupportEnv:
    def __init__(self):
        self.step_count = 0
        self.max_steps = 5
        self.issue_resolved = False

    def reset(self):
        self.step_count = 0
        self.issue_resolved = False
        return "Customer: Hi, I was charged twice for my subscription this month. Can you help me out?"

    def step(self, action: str):
        self.step_count += 1
        done = False
        reward = 0.0
        error = None
        obs = ""

        # Validate action (ensure no newlines since stdout must be single-line)
        if not isinstance(action, str):
            error = "Action must be a string"
            return obs, reward, True, error

        # Simulate customer logic based on agent action
        action_lower = action.lower()
        if "refund" in action_lower or "reverse the charge" in action_lower:
            obs = "Customer: Thank you so much! Really appreciate the swift refund."
            reward = 1.0  # Positive reward for resolving the issue correctly
            done = True
            self.issue_resolved = True
        elif "account number" in action_lower or "email address" in action_lower:
            obs = "Customer: My email is john.doe@example.com."
            reward = 0.0
        elif self.step_count >= self.max_steps:
            obs = "Customer: This is taking too long. I want to speak to a human."
            reward = -1.0 # Negative reward for failing to resolve in time
            done = True
        else:
            obs = "Customer: I don't think that helps my billing issue..."
            reward = 0.0
            
        return obs, reward, done, error

    def close(self):
        pass

# ==============================================================================
# 3. Main Inference Loop
# ==============================================================================
def run_inference():
    # Setup state
    task_name = "resolve-billing-issue"
    benchmark = "customer-support-v1"
    
    env = CustomerSupportEnv()
    obs = env.reset()
    
    # Must emit [START] at episode begin
    print(f"[START] task={task_name} env={benchmark} model={MODEL_NAME}")
    
    done = False
    rewards = []
    
    # Prompt context for the model
    messages = [
        {"role": "system", "content": "You are a customer support agent. Help the customer resolve their query effectively in as few steps as possible. If they need a refund, you can tell them you will 'process a refund' to resolve it."}
    ]
    
    try:
        while not done:
            # 1. Provide Observation to LLM
            messages.append({"role": "user", "content": obs})
            
            # 2. Get LLM Action
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    # Optional optimizations to stay within compute constraints
                    max_tokens=150,
                    temperature=0.3
                )
                action = response.choices[0].message.content.strip()
                
                # Format to remove newlines for strictly formatted stdout rules
                action = action.replace('\n', ' ').replace('\r', '')
            except Exception as e:
                action = "error"
                print(f"[STEP] step={env.step_count + 1} action=null reward=0.00 done=true error=\"{str(e)}\"")
                break
                
            # 3. Step Environment
            obs, reward, done, error = env.step(action)
            messages.append({"role": "assistant", "content": action})
            rewards.append(reward)
            
            # Formats conforming to specification
            action_str = repr(action) # Safe string representation
            error_str = repr(error) if error else "null"
            done_str = "true" if done else "false"
            
            # Must emit [STEP] per step, immediately after env.step()
            print(f"[STEP] step={env.step_count} action={action_str} reward={reward:.2f} done={done_str} error={error_str}")
            
    finally:
        # Wrap up cleanup
        env.close()
        
        # Calculate episode success metrics
        success_str = "true" if env.issue_resolved else "false"
        rewards_str = ",".join([f"{r:.2f}" for r in rewards]) if rewards else "0.00"
        
        # Must emit [END] after env.close(), always emitted
        print(f"[END] success={success_str} steps={env.step_count} rewards={rewards_str}")

if __name__ == "__main__":
    # Ensure it only runs if the file is imported directly to prevent execution on unintended imports
    run_inference()