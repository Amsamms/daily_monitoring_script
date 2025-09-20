import subprocess
import pyautogui
import time

# Path to Visual Studio Code executable and the notebook file
vscode_path = r'C:\Users\asabri\AppData\Local\Programs\Microsoft VS Code\Code.exe'
notebook_path = r'C:\Users\asabri\Jupitar notes\Daily Macro Trends\Daily_Macro_monitoring.ipynb'

# Properly quote each part if the path contains spaces
command = f'"{vscode_path}" "{notebook_path}"'

# Open the notebook in Visual Studio Code
subprocess.run(command, shell=True)

# Wait for Visual Studio Code to load (adjust the sleep time as necessary)
time.sleep(20)

# Open the command palette
pyautogui.hotkey('ctrl', 'shift', 'p')
time.sleep(2)

# Type "Run All Cells" in the command palette
pyautogui.write('Run All', interval=0.1)
pyautogui.press('enter')

# Wait for the notebook cells to complete execution (adjust the sleep time as necessary)
time.sleep(300)