import math
from abc import ABC, abstractmethod
from numpy.core.numeric import full

import torch
import onnxruntime
import os

from torch.cuda import device


class MuZeroNetwork:
    def __new__(cls, config):
        if config.network == "resnet":
            return MuZeroResidualNetwork(
                len(config.action_space), config.support_size, config.game_name
            )
        else:
            raise NotImplementedError(
                'The network parameter should be "resnet" for ONNX models.'
            )


def dict_to_cpu(dictionary):
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict


class AbstractNetwork(ABC, torch.nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def initial_inference(self, observation):
        pass

    @abstractmethod
    def recurrent_inference(self, encoded_state, action):
        pass

    def get_weights(self):
        return dict_to_cpu(self.state_dict())

    def set_weights(self, weights):
        self.load_state_dict(weights)


##################################
############# ResNet #############


def conv3x3(in_channels, out_channels, stride=1):
    return torch.nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )


# Residual block
class ResidualBlock(torch.nn.Module):
    def __init__(self, num_channels, stride=1):
        super().__init__()
        self.conv1 = conv3x3(num_channels, num_channels, stride)
        self.bn1 = torch.nn.BatchNorm2d(num_channels)
        self.conv2 = conv3x3(num_channels, num_channels)
        self.bn2 = torch.nn.BatchNorm2d(num_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.nn.functional.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += x
        out = torch.nn.functional.relu(out)
        return out


class MuZeroResidualNetwork(AbstractNetwork):
    def __init__(self, action_space_size, support_size, game_name):
        super().__init__()
        self.action_space_size = action_space_size
        self.full_support_size = 2 * support_size + 1

        model_path = f"./masters/onnx/{game_name}/"

        network_idents = ["rep", "dyn", "pre"]
        device_idents = ["cpu"]
        full_model_paths = [None, None, None]
        chosen_device = None

        if torch.cuda.is_available():
            device_idents.insert(0, "gpu")

        for device_ident in device_idents:
            print(f"Trying to use {device_ident.upper()}")
            for i, network_ident in enumerate(network_idents):
                full_model_path = f"{model_path}onnx_model_{network_ident}_net.quant_{device_ident}.onnx"

                if os.path.exists(full_model_path):
                    chosen_device = device_ident
                    full_model_paths[i] = full_model_path
                else:
                    print(f"{device_ident.upper()} available but no usable model found")
                    chosen_device = None
                    break
            if chosen_device:
                break

        if chosen_device:
            print(f"Model loaded in {chosen_device.upper()}")
            if chosen_device == "gpu":
                providers = ["CUDAExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
        else:
            print("No model found for available devices")

        self.rep_net_session = onnxruntime.InferenceSession(
            full_model_paths[0], providers=providers
        )

        self.dyn_net_session = onnxruntime.InferenceSession(
            full_model_paths[1], providers=providers
        )

        self.pre_net_session = onnxruntime.InferenceSession(
            full_model_paths[2], providers=providers
        )

    def prediction(self, encoded_state):
        # compute ONNX Runtime output prediction
        ort_inputs = {
            self.pre_net_session.get_inputs()[0].name: to_numpy(encoded_state)
        }
        ort_outs = self.pre_net_session.run(None, ort_inputs)
        policy, value = torch.tensor(ort_outs[0]), torch.tensor(ort_outs[1])
        return policy, value

    def representation(self, observation):
        # compute ONNX Runtime output prediction
        ort_inputs = {self.rep_net_session.get_inputs()[0].name: to_numpy(observation)}
        ort_outs = self.rep_net_session.run(None, ort_inputs)
        encoded_state = torch.tensor(ort_outs[0])

        return self._extracted_from_dynamics_8(encoded_state)

    def dynamics(self, encoded_state, action):
        # Stack encoded_state with a game specific one hot encoded action (See paper appendix Network Architecture)
        action_one_hot = (
            torch.ones(
                (
                    encoded_state.shape[0],
                    1,
                    encoded_state.shape[2],
                    encoded_state.shape[3],
                )
            )
            .to(action.device)
            .float()
        )
        action_one_hot = (
            action[:, :, None, None] * action_one_hot / self.action_space_size
        )
        x = torch.cat((encoded_state, action_one_hot), dim=1)

        # compute ONNX Runtime output prediction
        ort_inputs = {self.dyn_net_session.get_inputs()[0].name: to_numpy(x)}
        ort_outs = self.dyn_net_session.run(None, ort_inputs)
        next_encoded_state, reward = torch.tensor(ort_outs[0]), torch.tensor(
            ort_outs[1]
        )

        next_encoded_state_normalized = self._extracted_from_dynamics_8(next_encoded_state)

        return next_encoded_state_normalized, reward

    # TODO Rename this here and in `representation` and `dynamics`
    def _extracted_from_dynamics_8(self, arg0):
        min_encoded_state = arg0.view(-1, arg0.shape[1], arg0.shape[2] * arg0.shape[3]).min(2, keepdim=True)[0].unsqueeze(-1)

        max_encoded_state = arg0.view(-1, arg0.shape[1], arg0.shape[2] * arg0.shape[3]).max(2, keepdim=True)[0].unsqueeze(-1)

        scale_encoded_state = max_encoded_state - min_encoded_state
        scale_encoded_state[scale_encoded_state < 1e-05] += 1e-05
        return ((arg0 - min_encoded_state) / scale_encoded_state)

    def initial_inference(self, observation):
        encoded_state = self.representation(observation)
        policy_logits, value = self.prediction(encoded_state)
        # reward equal to 0 for consistency
        reward = torch.log(
            (
                torch.zeros(1, self.full_support_size)
                .scatter(1, torch.tensor([[self.full_support_size // 2]]).long(), 1.0)
                .repeat(len(observation), 1)
                .to(observation.device)
            )
        )
        return (
            value,
            reward,
            policy_logits,
            encoded_state,
        )

    def recurrent_inference(self, encoded_state, action):
        next_encoded_state, reward = self.dynamics(encoded_state, action)
        policy_logits, value = self.prediction(next_encoded_state)
        return value, reward, policy_logits, next_encoded_state


########### End ResNet ###########
##################################


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )
