import wandb
import pandas as pd
import yaml
import re
import csv
import os

def uploading(additional_step, step, config, name):
    FILENAME = additional_step
    FILENAME2 = step
    config_fie = config
    loaded_experiment_df = pd.DataFrame(FILENAME, columns=['Step', 'Skipped', 'Learning Rate', 'Momentum'])
    loaded_experiment_df2 = pd.DataFrame(FILENAME2, columns=['Steps','Loss','Iter Time (s)','Samples/sec'])

    print(loaded_experiment_df.head())
    with open(config_fie, 'r') as file:
        config_data = yaml.safe_load(file)

    # summaries = {
    #      "lm_loss":  1.795630E+00,
    #      "lm_loss_ppl": 6.023271E+00
    #  }

    PROJECT_NAME = "test"

    METRIC_COLS = ["Step", "Learning Rate"]
    METRIC_COLS2 = ["Steps", "Loss"]
    CONFIG_COLS = ["num-layers","hidden-size","num-attention-heads","seq-length","max-position-embeddings","norm","pos-emb","no-weight-tying","gpt_j_residual","output_layer_parallelism","include_bias_in_linear"]

    run_name = name

    metrics = {}
    for metric_col in METRIC_COLS:
        if metric_col == "Step":
            metrics[metric_col] = loaded_experiment_df[metric_col].tolist()
        elif metric_col == "Learning Rate":
            lr = []
            for lr_str in loaded_experiment_df[metric_col]:
                lr_float = float(lr_str.split(",")[0])
                lr.append(lr_float)
            metrics[metric_col] = lr

    config = {}

    for config_col in CONFIG_COLS:
            config[config_col] = config_data[config_col]

    for metric_col in METRIC_COLS2:
        if metric_col == "Steps":
            pass
        elif metric_col == "Loss":
            loss = []
            for loss_str in loaded_experiment_df2[metric_col]:
                loss_float = float(loss_str)
                loss.append(loss_float)
            metrics[metric_col] = loss

    print(metrics['Step'][1])


    run = wandb.init(
        project=PROJECT_NAME, name=run_name, config=config
    )

    wandb.define_metric("Step")
    # set all other train/ metrics to use this step
    wandb.define_metric("*", step_metric="Step")

    for step, lr, loss in zip(metrics['Step'], metrics['Learning Rate'], metrics['Loss']):
        print(step,lr,loss)
        wandb.log({"Step": step, "Learning Rate": lr , "Loss": loss})

    # run.summary.update(summaries)
    run.finish()

folder_path = "fp16_840M_no_bias"

def extract_number(filename):
    match = re.search(r'\.(\d+)$', filename)
    if match:
        return int(match.group(1))
    return -1

# Get a list of all files in the folder
files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and not f.endswith('.yml') and not f.endswith('ERROR')]

# Sort the files based on the last number in the filename
files.sort(key=extract_number)

# Open and print the contents of each file
total_additional_steps = []
total_steps = []

for filename in files:
    with open(os.path.join(folder_path, filename), 'r') as file:
        print(filename)
        log_output = file.read()

        step_pattern = re.compile(r"steps: (\d+) loss: ([\d.]+) iter time \(s\): ([\d.]+.\d+) samples/sec: ([\d.]+.\d+)")
        step_data = step_pattern.findall(log_output)

        additional_step_pattern = re.compile(r"step=(\d+), skipped=(\d+), lr=\[(.+)\], mom=\[(.+)\]")
        additional_step_data = additional_step_pattern.findall(log_output)

        additional_step_data = additional_step_data[1:]

        total_additional_steps.extend(additional_step_data)
        total_steps.extend(step_data)

yml_file_path = os.path.join(folder_path, folder_path + '.yml')

uploading(total_additional_steps,total_steps, yml_file_path, folder_path)



