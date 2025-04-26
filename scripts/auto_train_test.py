import os
import itertools
import subprocess
import re
import matplotlib.pyplot as plt

# 参数空间
model_names = ["CNN", "LSTM"]
epochs_list = [1]
lr_list = [2e-5, 1e-4]
num_layers_list = [1, 2]
dropout_list = [0.2, 0.5]
bidirectional_list = [False, True]

results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

def get_result_filename(model_name, epochs, lr, num_layers, dropout, bidirectional=None):
    fname = f"{model_name}-{epochs}-{lr:.6f}-{dropout}-{num_layers}"
    if model_name == "LSTM":
        fname += f"-{'bi' if bidirectional else 'uni'}"
    return os.path.join(results_dir, fname + ".txt")

for model_name in model_names:
    for epochs, lr, num_layers, dropout in itertools.product(epochs_list, lr_list, num_layers_list, dropout_list):
        if model_name == "CNN":
            result_file = get_result_filename(model_name, epochs, lr, num_layers, dropout)
            cmd = [
                "python", "/home/wuwen/python_project/nlp2/main.py", model_name,
                "--epochs", str(epochs),
                "--lr", str(lr),
                "--num_layers", str(num_layers),
                "--dropout", str(dropout),
                "--mode", "train"
            ]
            print(f"Running: {' '.join(cmd)}")
            with open(result_file, "w", encoding="utf-8") as fout:
                subprocess.run(cmd, stdout=fout, stderr=subprocess.STDOUT)
        elif model_name == "LSTM":
            for bidirectional in bidirectional_list:
                result_file = get_result_filename(model_name, epochs, lr, num_layers, dropout, bidirectional)
                cmd = [
                    "python", "/home/wuwen/python_project/nlp2/main.py", model_name,
                    "--epochs", str(epochs),
                    "--lr", str(lr),
                    "--num_layers", str(num_layers),
                    "--dropout", str(dropout),
                    "--mode", "train"
                ]
                if bidirectional:
                    cmd.append("--bidirectional")
                print(f"Running: {' '.join(cmd)}")
                with open(result_file, "w", encoding="utf-8") as fout:
                    subprocess.run(cmd, stdout=fout, stderr=subprocess.STDOUT)