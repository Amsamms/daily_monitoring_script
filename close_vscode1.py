import os
import signal
import subprocess

# Function to find and terminate VS Code process
def terminate_vscode():
    # Issue the 'taskkill' command to terminate VS Code processes
    os.system("taskkill /f /im Code.exe")

# Call the function to terminate VS Code
terminate_vscode()

print("Script to close Visual Studio Code has finished.")