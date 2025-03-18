import os
import subprocess

def generate_and_execute_slurm_job(
    python_start_script: str,
    account: str = "pengyu-lab", # "guest"
    partition: str = "pengyu-gpu", # "guest-gpu"
    job_name: str = "job",
    qos: str = "medium", # "low-gpu"
    time: str = "72:00:00", # "24:00:00"
    gpu: str = "",
    num: str = "",
    log_file_path: str = "./{job_name}.out",
    script_path:str ="./execute.sh"
):  
    script_template = """#!/bin/bash

#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --job-name={job_name}
#SBATCH --qos={qos}
#SBATCH --time={time}
#SBATCH --output={log_file_path}
{cpu_or_gpu}

# Set up env
source ~/.bashrc
conda activate llm

# Path to your executable
python {python_start_script}
"""

    cpu_or_gpu = "#SBATCH --gres=gpu:{gpu}:{num}".format(gpu=gpu, num=num) if gpu != "" \
        else "#SBATCH --cpus-per-task={num}".format(num=num)
    
    script_template = script_template.format(
        account=account,
        partition=partition,
        job_name=job_name,
        qos=qos,
        time=time,
        log_file_path=log_file_path.format(job_name=job_name),
        cpu_or_gpu=cpu_or_gpu,
        python_start_script=python_start_script
    )
    # Write the script to a file
    with open(script_path, 'w') as file:
        file.write(script_template)
    # Make the script executable
    os.chmod(script_path, 0o755)
    
    print(f"Created execute.sh at {os.path.abspath(script_path)}")
    
    # Submit the script using sbatch
    result = subprocess.run(['sbatch', script_path], capture_output=True, text=True)

    if result.returncode == 0:
        job_id = result.stdout.split()[-1]
        print(f"[Job ID: {job_id}] Job submitted successfully!")
        return job_name
    
    print("Error submitting job:")
    print(result.stderr)
    
    return None