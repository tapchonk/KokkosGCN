import subprocess
import time

# Define the command
commands = ["./remoterun.sh ./gnnAlgorithm.cuda -Train 800 -Test 200 -Hidden 47 -E 100 -C 3 -LR 1.0 -AT 100.0",
             "./remoterun.sh ./gnnAlgorithm.cuda -Train 800 -Test 200 -Hidden 47 -E 100 -C 3 -LR 1.0 -AT 100.0",
             "./remoterun.sh ./gnnAlgorithm.cuda -Train 3200 -Test 800 -Hidden 47 -E 100 -C 3 -LR 1.0 -AT 100.0",
             "./remoterun.sh ./gnnAlgorithm.cuda -Train 6400 -Test 1600 -Hidden 47 -E 100 -C 3 -LR 1.0 -AT 100.0",
             "./remoterun.sh ./gnnAlgorithm.cuda -Train 12800 -Test 3200 -Hidden 47 -E 100 -C 3 -LR 1.0 -AT 100.0",
             "./remoterun.sh ./gnnAlgorithm.cuda -Train 25600 -Test 6400 -Hidden 47 -E 100 -C 3 -LR 1.0 -AT 100.0",
             "./remoterun.sh ./gnnAlgorithm.cuda -Train 51200 -Test 12800 -Hidden 47 -E 100 -C 3 -LR 1.0 -AT 100.0",
             "./remoterun.sh ./gnnAlgorithm.cuda -Train 102400 -Test 25600 -Hidden 47 -E 100 -C 3 -LR 1.0 -AT 100.0",
             "./remoterun.sh ./gnnAlgorithm.cuda -Train 204800 -Test 51200 -Hidden 47 -E 100 -C 3 -LR 1.0 -AT 100.0",
             "./remoterun.sh ./gnnAlgorithm.cuda -Train 409600 -Test 102400 -Hidden 47 -E 100 -C 3 -LR 1.0 -AT 100.0",
             "./remoterun.sh ./gnnAlgorithm.cuda -Train 819200 -Test 204800 -Hidden 47 -E 100 -C 3 -LR 1.0 -AT 100.0",
             "./remoterun.sh ./gnnAlgorithm.cuda -Train 1638400 -Test 409600 -Hidden 47 -E 100 -C 3 -LR 1.0 -AT 100.0"]

queuecommand = "squeue"

for i in range(0, 12):
    # Run the command
    print("Executing command")
    for _ in range(0, 5):
        subprocess.run(commands[i], shell=True)
        time.sleep(1)
    
    print("Command executed successfully")
    print("Checking queue status")
    while True:
        # get the output of the command
        output = subprocess.check_output(queuecommand, shell=True)
        # count number of times 2145461 appears in the output
        count = output.count(b"2145461")/2
        print(f"Number of 2145461 queues: {count}")
        if count > 10:
            print("2145461 queue found")
        else:
            print("2145461 queue not found")
            break
        time.sleep(15)
    