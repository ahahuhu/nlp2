import os
import itertools
import subprocess

# 参数空间
model_names = ["CNN", "LSTM"]
epochs_list = [20]
lr_list = [2e-5]
num_layers_list = [1, 3]
dropout_list = [0.0, 0.2, 0.5]
bidirectional_list = [False, True]

results_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)),"results"))
print(results_dir)
os.makedirs(results_dir, exist_ok=True)

def get_result_filename(model_name, epochs, lr, num_layers, dropout, bidirectional=None):
    fname = f"{model_name}-{epochs}-{lr:.6f}-{dropout}-{num_layers}"
    if model_name == "LSTM":
        fname += f"-{'bi' if bidirectional else 'uni'}"
    return os.path.join(results_dir, fname + ".txt")

FILE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for model_name in model_names:
    for epochs, lr, num_layers, dropout in itertools.product(epochs_list, lr_list, num_layers_list, dropout_list):
        if model_name == "CNN":
            result_file = get_result_filename(model_name, epochs, lr, num_layers, dropout)
            cmd = [
                "python", f"{FILE_PATH}/main.py", model_name,
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
                    "python", f"{FILE_PATH}/main.py", model_name,
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