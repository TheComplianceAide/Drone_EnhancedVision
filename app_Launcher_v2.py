import tkinter as tk
import os
import subprocess
import psutil

class App:
    def __init__(self, master, path):
        self.master = master
        self.path = path
        self.master.title("Drone CV Vision")
        self.master.geometry("1100x250")

        # Create a container frame for the buttons
        self.container = tk.Frame(self.master)
        self.container.pack(fill="both", expand=True)

        # Create a variable to store the subprocess object
        self.process = None

        # Create a variable to store the process ID
        self.pid = None

        # Create a button to launch the selected script
        self.launch_btn = tk.Button(self.master, text="Launch script", width=20, height=2, command=self.launch_script)
        self.launch_btn.pack(side="left", padx=10, pady=10)

        # Create a button to kill the running script
        self.kill_btn = tk.Button(self.master, text="Kill script", width=20, height=2, state="disabled", command=self.kill_script)
        self.kill_btn.pack(side="right", padx=10, pady=10)

        self.create_buttons()

    def create_buttons(self):
        scripts = os.listdir(self.path)
        row = 0
        col = 0
        max_cols = 3 # Set the maximum number of columns here
        for i, script in enumerate(scripts):
            if script.endswith('.py') and script[0] == '_':  # Only include scripts where first character is '_'
                btn = tk.Button(self.container, text=script, width=40, height=2, command=lambda script=script: self.select_script(script))
                btn.grid(row=row, column=col, padx=5, pady=5)

                col += 1
                if col == max_cols:
                    col = 0
                    row += 1

    def select_script(self, script):
        self.script = script
        self.launch_btn.config(state="normal")

    def launch_script(self):
        if self.process is not None:
            # A script is already running, so we need to kill it first
            self.kill_script()

        # Launch the selected script as a subprocess
        self.process = subprocess.Popen(['python', os.path.join(self.path, self.script)])
        self.pid = self.process.pid

        # Disable the "Launch script" button and enable the "Kill script" button
        self.launch_btn.config(state="disabled")
        self.kill_btn.config(state="normal")

    def kill_script(self):
        if self.pid is not None:
            try:
                # Terminate the subprocess using the process ID
                process = psutil.Process(self.pid)
                process.terminate()

                # Wait for the process to terminate
                process.wait(timeout=1)

            except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                pass

            # Reset the subprocess variable and button states
            self.process = None
            self.pid = None
            self.launch_btn.config(state="normal")
            self.kill_btn.config(state="disabled")

if __name__ == '__main__':
    root = tk.Tk()
    app = App(root, r'C:\temp\Drone_CV_Vision')
    root.mainloop()
