import torch
import os
import models
import config
import numpy as np
import onnx
import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType


def main():
    game_name = "tictactoe"
    checkpoint_path = f"masters/{game_name}/model.checkpoint"
    _config = config.Config(game_name)
    _config.game_filename = game_name
    torch_model = models.MuZeroNetwork(_config)

    n_channels = _config.channels
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # for name, param in torch_model.named_parameters():
    #     print(name, param.size())

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        print(f"\nUsing checkpoint from {checkpoint_path}")
    else:
        print(f"\nThere is no model saved in {checkpoint_path}.")
    torch_model.set_weights(checkpoint["weights"])
    torch_model.to(torch.device(device))

    # Extract models
    torch_model_rep_net = torch_model.representation_network.module
    torch_model_rep_net.eval()
    torch_model_dyn_net = torch_model.dynamics_network.module
    torch_model_dyn_net.eval()
    torch_model_pre_net = torch_model.prediction_network.module
    torch_model_pre_net.eval()

    # Export Representation Network Model
    onnx_model_rep_net_path = f"masters/onnx/{game_name}/onnx_model_rep_net.onnx"
    export_model(torch_model_rep_net, onnx_model_rep_net_path, (1, 3, 3, 3), device)

    # Export Dynamics Network Model
    onnx_model_dyn_net_path = f"masters/onnx/{game_name}/onnx_model_dyn_net.onnx"
    export_model(
        torch_model_dyn_net, onnx_model_dyn_net_path, (1, n_channels + 1, 3, 3), device
    )

    # Export Prediction Network Model
    onnx_model_pre_net_path = f"masters/onnx/{game_name}/onnx_model_pre_net.onnx"
    export_model(
        torch_model_pre_net, onnx_model_pre_net_path, (1, n_channels, 3, 3), device
    )

    # Quantize models
    onnx_model_rep_net_path = quantize_model(onnx_model_rep_net_path)
    onnx_model_dyn_net_path = quantize_model(onnx_model_dyn_net_path)
    onnx_model_pre_net_path = quantize_model(onnx_model_pre_net_path)

    # Optimize models
    onnx_model_rep_net_path = optimize_by_onnxruntime(
        onnx_model_rep_net_path, device == "cuda"
    )
    onnx_model_dyn_net_path = optimize_by_onnxruntime(
        onnx_model_dyn_net_path, device == "cuda"
    )
    onnx_model_pre_net_path = optimize_by_onnxruntime(
        onnx_model_pre_net_path, device == "cuda"
    )


def export_model(torch_model, onnx_model_path, input_shape, device):
    dummy_input = torch.randn(input_shape, requires_grad=True, device=device)
    torch_out = torch_model(dummy_input)
    torch.onnx.export(torch_model, dummy_input, onnx_model_path, opset_version=11)
    if device == "gpu":
        providers = ["CUDAExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    # Evaluate model
    evaluate_model(onnx_model_path, torch_out, dummy_input, providers)


def evaluate_model(onnx_model_path, torch_out, dummy_input, providers):
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession(onnx_model_path, providers=providers)

    def to_numpy(tensor):
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    ort_outs = ort_session.run(None, ort_inputs)

    if type(torch_out) == torch.Tensor:
        # compare ONNX Runtime and PyTorch results
        np.testing.assert_allclose(
            to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05
        )
        print(
            f"Exported model ({onnx_model_path}) has been tested with ONNXRuntime, and the result looks good!"
        )

    elif type(torch_out) == tuple:
        for i, _ in enumerate(torch_out):
            # compare ONNX Runtime and PyTorch results
            np.testing.assert_allclose(
                to_numpy(torch_out[i]), ort_outs[i], rtol=1e-03, atol=1e-05
            )
            print(
                f"Exported model ({onnx_model_path}[{i}]) has been tested with ONNXRuntime, and the result looks good!"
            )


def optimize_by_onnxruntime(onnx_model_path, use_gpu, optimized_model_path=None):
    """
    Use onnxruntime to optimize model.
    Args:
        onnx_model_path (str): the path of input onnx model.
        use_gpu (bool): whether the optimized model is targeted to run in GPU.
        optimized_model_path (str or None): the path of optimized model.
    Returns:
        optimized_model_path (str): the path of optimized model
    """

    if use_gpu and "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
        print("There is no gpu for onnxruntime to do optimization.")
        return onnx_model_path

    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    if optimized_model_path is None:
        path_prefix = onnx_model_path[:-5]  # remove .onnx suffix
        optimized_model_path = f'{path_prefix}_{"gpu" if use_gpu else "cpu"}.onnx'

    sess_options.optimized_model_filepath = optimized_model_path

    if not use_gpu:
        session = onnxruntime.InferenceSession(
            onnx_model_path, sess_options, providers=["CPUExecutionProvider"]
        )
    else:
        session = onnxruntime.InferenceSession(onnx_model_path, sess_options)
        assert (
            "CUDAExecutionProvider" in session.get_providers()
        )  # Make sure there is GPU

    assert os.path.exists(optimized_model_path) and os.path.isfile(optimized_model_path)
    print(f"Save optimized model by onnxruntime to {optimized_model_path}")
    os.remove(onnx_model_path)
    os.remove(f"{path_prefix[:-6]}-opt.onnx")
    return optimized_model_path


def quantize_model(onnx_model_path, quantized_model_path=None):
    if quantized_model_path is None:
        path_prefix = onnx_model_path[:-5]  # remove .onnx suffix
        quantized_model_path = f"{path_prefix}.quant.onnx"
    quantize_dynamic(
        onnx_model_path, quantized_model_path, weight_type=QuantType.QUInt8
    )
    os.remove(onnx_model_path)
    return quantized_model_path


if __name__ == "__main__":
    main()
