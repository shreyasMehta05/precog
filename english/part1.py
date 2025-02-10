#!/usr/bin/env python3
import subprocess
import os
from colorama import init, Fore, Style
# import time to measure the time taken to run the scripts
import time
# Initialize colorama for colored terminal output.
init(autoreset=True)

def run_script(script_path):
    """
    Runs the given script with python3 and returns its standard output and error.
    """
    try:
        result = subprocess.run(
            ["python3", script_path],
            capture_output=True,  # Capture both stdout and stderr.
            text=True,            # Decode output as text.
            check=True            # Raise CalledProcessError on non-zero exit.
        )
        return result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        # Capture output even if there is an error.
        return e.stdout, e.stderr

def main():
    # Define the directory and list of scripts.
    scripts_dir = "scripts"
    scripts = [
        "tokenizer.py",
        "cooccurrence_builder.py",
        "matrix_reducer.py",
        "enhanced_evaluation.py",
        "trends.py"
    ]
    
    
    output_file = "TerminalOutputs.txt"
    
    
    # time
    start_time = time.time()
    with open(output_file, "w") as outfile:
        for script in scripts:
            script_path = os.path.join(scripts_dir, script)
            
            # Prepare header text.
            header = f"Running {script_path}\n" + "=" * 80 + "\n"
            outfile.write(header)
            
            # Print header in cyan.
            print(f"{Fore.CYAN}Running {script_path}{Style.RESET_ALL}")
            print("=" * 80)
            
            # Run the script.
            stdout, stderr = run_script(script_path)
            
            # Process standard output.
            if stdout:
                stdout_header = "Standard Output:\n"
                outfile.write(stdout_header + stdout + "\n")
                print(f"{Fore.GREEN}{stdout_header}{Style.RESET_ALL}", end="")
                print(stdout)
            
            # Process standard error.
            if stderr:
                stderr_header = "Standard Error:\n"
                outfile.write(stderr_header + stderr + "\n")
                print(f"{Fore.RED}{stderr_header}{Style.RESET_ALL}", end="")
                print(stderr)
            
            # Separator between script outputs.
            separator = "=" * 80 + "\n\n"
            outfile.write(separator)
            print(separator)
            print(f"{Fore.MAGENTA}Finished running {script}{Style.RESET_ALL}\n")
    # end
    end_time = time.time()
    print(f"{Fore.GREEN}All scripts executed in {end_time - start_time:.2f} seconds{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}All scripts executed. Output saved to {output_file}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
