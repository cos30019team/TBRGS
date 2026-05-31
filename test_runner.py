import csv
import subprocess
import os
import re

# Standard labels used in your assignment
ALGORITHMS = ["BFS", "DFS", "CUS1", "CUS2", "ASTAR", "GBFS"]
TEST_CASES = [f"test{i}.txt" for i in range(1, 16)]
TC_FOLDER = "TC"

def run_tests():
    results = []
    headers = ["Test Case", "Algorithm", "Goal Found", "Nodes Created", "Path Cost", "Path Length", "Status"]
    
    if not os.path.exists(TC_FOLDER):
        print(f"Error: Folder '{TC_FOLDER}' not found.")
        return

    for tc in TEST_CASES:
        file_path = os.path.join(TC_FOLDER, tc)
        if not os.path.exists(file_path):
            continue
            
        print(f"--- Running {tc} ---")
        
        for alg in ALGORITHMS:
            try:
                # Execute search.py and capture stdout
                process = subprocess.run(
                    ["python3", "search.py", file_path, alg],
                    capture_output=True,
                    text=True,
                    timeout=10 # Prevents infinite loops in DFS
                )
                
                output = process.stdout.strip()
                lines = output.split('\n')
                
                # Logic to find the 'Summary' line (Goal NodeCount Cost)
                # This ignores 'Hello' messages and only looks for the line with 2 or 3 numbers
                summary_data = None
                path_str = "No Path"
                
                for line in lines:
                    parts = line.split()
                    # Check if line looks like: 'GoalNode Count [Cost]'
                    if len(parts) >= 2 and parts[1].isdigit():
                        summary_data = parts
                    elif "->" in line:
                        path_str = line

                if summary_data:
                    goal = summary_data[0]
                    nodes_created = summary_data[1]
                    # Attempt to get cost if your script outputs it, else default to 'N/A'
                    path_len = len(path_str.split(" -> ")) if "->" in path_str else 0
                    
                    results.append([tc, alg, goal, nodes_created,  path_len, "PASS"])
                    print(f"  [OK] {alg}: Nodes={nodes_created}")
                else:
                    results.append([tc, alg, "No Goal", "0", "0", "0", "NO_PATH"])
                    print(f"  [!] {alg}: No path found.")

            except subprocess.TimeoutExpired:
                results.append([tc, alg, "Timeout", "0", "0", "0", "FAIL"])
                print(f"  [X] {alg}: Timed out (Check for infinite loops)")
            except Exception as e:
                results.append([tc, alg, "Error", "0", "0", "0", "ERROR"])
                print(f"  [X] {alg}: Execution Error")

    # Save to CSV
    with open("testing_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(results)
    
    print("\nDone! Open 'testing_results.csv' to analyze your algorithm insights.")

if __name__ == "__main__":
    run_tests()