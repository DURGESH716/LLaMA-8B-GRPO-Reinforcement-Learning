import json
import os
import glob
import pandas as pd
import re

def extract_score(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        task_res = data.get("results", {}).get("mbpp_instruct", {})
        # Looks for the specific key found in your JSON logs
        return task_res.get("pass_at_1,extract_code") or task_res.get("pass_at_1") or 0.0
    except:
        return 0.0

def get_checkpoint_name(filename):
    if 'base' in filename.lower():
        return 'base'
    
    # Improved regex: Find numbers that are NOT preceded by '0shot'
    # Or simply: look for numbers that are 3 digits or more (500, 1000, etc.)
    # or the specific '5760'
    match = re.findall(r'\d+', filename)
    if match:
        # If '0shot' is in the filename, the first match might be '0'. 
        # We want the one that isn't '0' or is the largest.
        nums = [n for n in match if n != '0']
        return nums[0] if nums else '0'
    return filename

def main():
    all_data = []
    
    # Mapping the folders
    shot_map = {'0shot': '0shot', '3shot': '3shot'}
    
    for folder, shot_type in shot_map.items():
        path = os.path.join("eval_results", folder, "*.json")
        for f in glob.glob(path):
            ckpt = get_checkpoint_name(os.path.basename(f))
            score = extract_score(f)
            all_data.append({'Checkpoint': ckpt, 'Type': shot_type, 'Score': score})

    df = pd.DataFrame(all_data)
    
    # Pivot: Index is Checkpoint, Columns are 0shot/3shot
    summary = df.pivot_table(index='Checkpoint', columns='Type', values='Score', aggfunc='max')

    # Logical Sorting
    def sort_key(idx):
        if idx == 'base': return -1
        try:
            return int(idx)
        except:
            return 99999
    
    sorted_indices = sorted(summary.index, key=sort_key)
    summary = summary.reindex(sorted_indices)

    # Calculate Improvement relative to 3-shot base
    if '3shot' in summary.columns and 'base' in summary.index:
        b_val = summary.loc['base', '3shot']
        summary['Improvement (%)'] = ((summary['3shot'] - b_val) / b_val) * 100

    print("\n========================================")
    print("      GRPO PERFORMANCE SUMMARY")
    print("========================================\n")
    print(summary.to_string(float_format=lambda x: "{:.4f}".format(x)))
    
    os.makedirs("results", exist_ok=True)
    summary.to_csv("results/final_comparison.csv")
    
    # Save a copy with NaNs filled for the plotter
    summary.interpolate(method='linear').ffill().bfill().to_csv("results/plot_data.csv")
    
    print(f"\nResults unified! Peak: {summary['3shot'].max():.4f}")

if __name__ == "__main__":
    main()