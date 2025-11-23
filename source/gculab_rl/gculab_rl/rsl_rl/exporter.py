# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import os

import torch


def export_policy_as_jit(policy: object, normalizer: object | None, path: str, filename="policy.pt"):
    """Export policy into a Torch JIT file.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported JIT file. Defaults to "policy.pt".
    """
    policy_exporter = _TorchPolicyExporter(policy, normalizer)
    policy_exporter.export(path, filename)


def export_policy_as_onnx(
    policy: object, path: str, normalizer: object | None = None, filename="policy.onnx", verbose=False
):
    """Export policy into a Torch ONNX file.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxPolicyExporter(policy, normalizer, verbose)
    policy_exporter.export(path, filename)


"""
Helper Classes - Private.
"""


class _TorchPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into JIT file."""

    def __init__(self, policy, normalizer=None):
        super().__init__()
        self.is_recurrent = policy.is_recurrent
        # copy policy parameters
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_a.rnn)
        elif hasattr(policy, "student"):
            self.actor = copy.deepcopy(policy.student)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_s.rnn)
        else:
            raise ValueError("Policy does not have an actor/student module.")
        # set up recurrent network
        if self.is_recurrent:
            self.rnn.cpu()
            self.rnn_type = type(self.rnn).__name__.lower()  # 'lstm' or 'gru'
            self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
            if self.rnn_type == "lstm":
                self.register_buffer("cell_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
                self.forward = self.forward_lstm
                self.reset = self.reset_memory
            elif self.rnn_type == "gru":
                self.forward = self.forward_gru
                self.reset = self.reset_memory
            else:
                raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
            # Check if this is a Conv2D network and normalizer has different input size
            self.is_conv2d = hasattr(self.actor, "image_input_shape")
            if self.is_conv2d:
                # For Conv2D networks, the normalizer was trained with only proprioceptive obs
                self.proprio_obs_size = normalizer._mean.shape[1]  # Get the size from normalizer
                self.image_obs_size = torch.prod(torch.tensor(self.actor.image_input_shape)).item()
        else:
            self.normalizer = torch.nn.Identity()
            self.is_conv2d = False

    def forward_lstm(self, x):
        if self.is_conv2d:
            # For Conv2D networks, only normalize the proprioceptive part
            proprio_obs = x[:, : self.proprio_obs_size]
            image_obs = x[:, self.proprio_obs_size :]
            normalized_proprio = self.normalizer(proprio_obs)
            # RNN processes full observations (proprio + image)
            rnn_input = torch.cat([normalized_proprio, image_obs], dim=1)
            rnn_output, (h, c) = self.rnn(rnn_input.unsqueeze(0), (self.hidden_state, self.cell_state))
            self.hidden_state[:] = h
            self.cell_state[:] = c
            rnn_output = rnn_output.squeeze(0)
            # For Conv2D networks, actor expects [rnn_output, image_obs]
            actor_input = torch.cat([rnn_output, image_obs], dim=1)
        else:
            rnn_input = self.normalizer(x)
            rnn_output, (h, c) = self.rnn(rnn_input.unsqueeze(0), (self.hidden_state, self.cell_state))
            self.hidden_state[:] = h
            self.cell_state[:] = c
            rnn_output = rnn_output.squeeze(0)
            actor_input = rnn_output
        
        return self.actor(actor_input)

    def forward_gru(self, x):
        if self.is_conv2d:
            # For Conv2D networks, only normalize the proprioceptive part
            proprio_obs = x[:, : self.proprio_obs_size]
            image_obs = x[:, self.proprio_obs_size :]
            normalized_proprio = self.normalizer(proprio_obs)
            # RNN processes full observations (proprio + image)
            rnn_input = torch.cat([normalized_proprio, image_obs], dim=1)
            rnn_output, h = self.rnn(rnn_input.unsqueeze(0), self.hidden_state)
            self.hidden_state[:] = h
            rnn_output = rnn_output.squeeze(0)
            # For Conv2D networks, actor expects [rnn_output, image_obs]
            actor_input = torch.cat([rnn_output, image_obs], dim=1)
        else:
            rnn_input = self.normalizer(x)
            rnn_output, h = self.rnn(rnn_input.unsqueeze(0), self.hidden_state)
            self.hidden_state[:] = h
            rnn_output = rnn_output.squeeze(0)
            actor_input = rnn_output
        
        return self.actor(actor_input)

    def forward(self, x):
        if self.is_conv2d:
            # For Conv2D networks, only normalize the proprioceptive part
            proprio_obs = x[:, : self.proprio_obs_size]
            image_obs = x[:, self.proprio_obs_size :]
            normalized_proprio = self.normalizer(proprio_obs)
            x = torch.cat([normalized_proprio, image_obs], dim=1)
        else:
            x = self.normalizer(x)
        return self.actor(x)

    @torch.jit.export
    def reset(self):
        pass

    def reset_memory(self):
        self.hidden_state[:] = 0.0
        if hasattr(self, "cell_state"):
            self.cell_state[:] = 0.0

    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


class _OnnxPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into ONNX file."""

    def __init__(self, policy, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.is_recurrent = policy.is_recurrent
        # copy policy parameters
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_a.rnn)
        elif hasattr(policy, "student"):
            self.actor = copy.deepcopy(policy.student)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_s.rnn)
        else:
            raise ValueError("Policy does not have an actor/student module.")
        # set up recurrent network
        if self.is_recurrent:
            self.rnn.cpu()
            self.rnn_type = type(self.rnn).__name__.lower()  # 'lstm' or 'gru'
            if self.rnn_type == "lstm":
                self.forward = self.forward_lstm
            elif self.rnn_type == "gru":
                self.forward = self.forward_gru
            else:
                raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
            # Check if this is a Conv2D network and normalizer has different input size
            self.is_conv2d = hasattr(self.actor, "image_input_shape")
            if self.is_conv2d:
                # For Conv2D networks, the normalizer was trained with only proprioceptive obs
                self.proprio_obs_size = normalizer._mean.shape[1]  # Get the size from normalizer
                self.image_obs_size = torch.prod(torch.tensor(self.actor.image_input_shape)).item()
        else:
            self.normalizer = torch.nn.Identity()
            self.is_conv2d = False

    def forward_lstm(self, x_in, h_in, c_in):
        if self.is_conv2d:
            # For Conv2D networks, only normalize the proprioceptive part
            proprio_obs = x_in[:, : self.proprio_obs_size]
            image_obs = x_in[:, self.proprio_obs_size :]
            normalized_proprio = self.normalizer(proprio_obs)
            # RNN processes full observations (proprio + image)
            rnn_input = torch.cat([normalized_proprio, image_obs], dim=1)
            rnn_output, (h, c) = self.rnn(rnn_input.unsqueeze(0), (h_in, c_in))
            rnn_output = rnn_output.squeeze(0)
            # For Conv2D networks, actor expects [rnn_output, image_obs]
            actor_input = torch.cat([rnn_output, image_obs], dim=1)
        else:
            rnn_input = self.normalizer(x_in)
            rnn_output, (h, c) = self.rnn(rnn_input.unsqueeze(0), (h_in, c_in))
            rnn_output = rnn_output.squeeze(0)
            actor_input = rnn_output
        
        return self.actor(actor_input), h, c

    def forward_gru(self, x_in, h_in):
        if self.is_conv2d:
            # For Conv2D networks, only normalize the proprioceptive part
            proprio_obs = x_in[:, : self.proprio_obs_size]
            image_obs = x_in[:, self.proprio_obs_size :]
            normalized_proprio = self.normalizer(proprio_obs)
            # RNN processes full observations (proprio + image)
            rnn_input = torch.cat([normalized_proprio, image_obs], dim=1)
            rnn_output, h = self.rnn(rnn_input.unsqueeze(0), h_in)
            rnn_output = rnn_output.squeeze(0)
            # For Conv2D networks, actor expects [rnn_output, image_obs]
            actor_input = torch.cat([rnn_output, image_obs], dim=1)
        else:
            rnn_input = self.normalizer(x_in)
            rnn_output, h = self.rnn(rnn_input.unsqueeze(0), h_in)
            rnn_output = rnn_output.squeeze(0)
            actor_input = rnn_output
        
        return self.actor(actor_input), h

    def forward(self, x):
        if self.is_conv2d:
            # For Conv2D networks, only normalize the proprioceptive part
            proprio_obs = x[:, : self.proprio_obs_size]
            image_obs = x[:, self.proprio_obs_size :]
            normalized_proprio = self.normalizer(proprio_obs)
            x = torch.cat([normalized_proprio, image_obs], dim=1)
        else:
            x = self.normalizer(x)
        return self.actor(x)

    def export(self, path, filename):
        self.to("cpu")
        self.eval()
        if self.is_recurrent:
            obs = torch.zeros(1, self.rnn.input_size)
            h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)

            if self.rnn_type == "lstm":
                c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
                torch.onnx.export(
                    self,
                    (obs, h_in, c_in),
                    os.path.join(path, filename),
                    export_params=True,
                    opset_version=11,
                    verbose=self.verbose,
                    input_names=["obs", "h_in", "c_in"],
                    output_names=["actions", "h_out", "c_out"],
                    dynamic_axes={},
                )
            elif self.rnn_type == "gru":
                torch.onnx.export(
                    self,
                    (obs, h_in),
                    os.path.join(path, filename),
                    export_params=True,
                    opset_version=11,
                    verbose=self.verbose,
                    input_names=["obs", "h_in"],
                    output_names=["actions", "h_out"],
                    dynamic_axes={},
                )
            else:
                raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")
        else:
            if self.is_conv2d:
                # For Conv2D networks, use the full observation size (proprio + image)
                input_size = self.proprio_obs_size + self.image_obs_size
            else:
                input_size = getattr(self.actor, "input_dim", None)
                if input_size is None:
                    input_size = self.actor[0].in_features
            obs = torch.zeros(1, input_size)
            torch.onnx.export(
                self,
                obs,
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs"],
                output_names=["actions"],
                dynamic_axes={},
            )
