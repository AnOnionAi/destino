import subprocess

game_name = "chess"
device = "gpu"
destination_path = f"./masters/onnx/{game_name}/"

onnx_model_dyn = "https://storage.googleapis.com/"

cmd = ["wget", onnx_model_dyn, "-P", destination_path]
process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
process.communicate()

onnx_model_pre = "https://storage.googleapis.com/"

cmd = ["wget", onnx_model_pre, "-P", destination_path]
process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
process.communicate()

onnx_model_rep = "https://storage.googleapis.com/"

cmd = ["wget", onnx_model_rep, "-P", destination_path]
process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
process.communicate()
