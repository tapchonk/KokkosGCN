import os
import subprocess

# Define function to process chunks
def process_chunk(chunk):
    avg_val = sum(chunk) / len(chunk)
    max_val = max(chunk)
    min_val = min(chunk)
    diff_max = max_val - avg_val
    diff_min = avg_val - min_val
    return diff_max, diff_min, avg_val

# Enter directories and run command
# directories = ["1__CORE", "2__CORE", "4__CORE", "8__CORE", "16__CORE", "32__CORE"]
directories = ["1", "2","4", "8", "16", "32", "48_GPU"]

for directory in directories:
    os.chdir(directory)
    output = subprocess.run(['awk \'/Back/{gsub(/[^0-9.]/,"",$0); print $0}\' *.out'], capture_output=True, text=True, shell=True)
    lines = output.stdout.strip().split('\n')
    chunks = [list(map(float, lines[i:i+5])) for i in range(0, len(lines), 5)]
    i = 0
    for chunk in chunks:
        i = i + 1
        diff_max, diff_min, avg_val = process_chunk(chunk)
        print(f"({i * 20}, {avg_val:.6f}) +- (-{diff_max:.6f}, {diff_min:.6f})")
        #print(f"Directory: {directory}, Avg: {avg_val:.6f}, Diff from Max: {diff_max:.6f}, Diff from Min: {diff_min:.6f}")
        #(20,  0.806239) +- (-100.0, 100.0)
    os.chdir("..")
