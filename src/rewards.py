import re

def format_reward_func(completions, **kwargs):
    """Reward for correct <think> and <answer> tag structure."""
    rewards = []
    for completion in completions:
        content = completion[0]["content"]
        # Regex to ensure both opening and closing tags for think and answer exist
        pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
        if re.search(pattern, content, re.DOTALL):
            rewards.append(0.2) 
        else:
            rewards.append(0.0)
    return rewards

def efficiency_reward_func(completions, **kwargs):
    """Penalizes answers that are cut off (no </answer> tag)."""
    rewards = []
    for completion in completions:
        content = completion[0]["content"]
        # If the model hits the max_completion_length, it won't have the closing tag
        if "</answer>" not in content:
            rewards.append(-0.5) # Strong negative signal to stop rambling
        else:
            # Small bonus if the model is concise (under 1200 characters)
            rewards.append(0.1 if len(content) < 1200 else 0.0)
    return rewards

def code_reward_func(completions, answer, **kwargs):
    rewards = []
    for completion, expected_test in zip(completions, answer):
        content = completion[0]["content"]
        code_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        
        if not code_match:
            rewards.append(0.0)
            continue
            
        code = code_match.group(1).strip()
        
        # 1. Partial Reward for defining a function (0.1)
        if "def " in code:
            # 2. Even more credit if it matches the function name in the test
            # Extracts function name from "assert find_volume(5) == 125"
            func_name_match = re.search(r"assert (\w+)\(", expected_test)
            if func_name_match and func_name_match.group(1) in code:
                rewards.append(0.4) # Huge hint: "You're on the right track!"
            else:
                rewards.append(0.1)
        else:
            rewards.append(0.0)
            
        # 3. Full Reward for passing (override previous partial)
        try:
            exec_globals = {}
            exec(code, exec_globals)
            exec(expected_test, exec_globals)
            rewards[-1] = 1.0 
        except:
            pass
    return rewards