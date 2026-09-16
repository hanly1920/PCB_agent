import json

# Load analysis results
with open('analysis_results.json', 'r') as f:
    results = json.load(f)

# Filter trajectories without overlaps
clean_trajectories = [r['trajectory'] for r in results if not r['has_overlaps']]

print(f"Found {len(clean_trajectories)} clean trajectories out of {len(results)} total")
print("Clean trajectories:")
for traj in clean_trajectories:
    print(f"  {traj}")

# Save clean trajectory list
with open('clean_trajectories.txt', 'w') as f:
    for traj in clean_trajectories:
        f.write(f"{traj}\n")

print(f"\nSaved clean trajectory list to clean_trajectories.txt")