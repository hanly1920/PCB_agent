import glob
import json
import os

def main():
    base = os.path.join('dta', 'expert_traj (2)', 'expert_traj')
    pattern = os.path.join(base, '**', '*_layout.json')
    files = glob.glob(pattern, recursive=True)
    max_count = 0
    max_path = None
    total = 0
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                count = len(data.keys())
            elif isinstance(data, list):
                count = len(data)
            else:
                count = 0
            total += 1
            if count > max_count:
                max_count = count
                max_path = f
        except Exception:
            continue

    print('files_scanned:', total)
    print('max_count:', max_count)
    print('max_path:', max_path)

if __name__ == '__main__':
    main()
