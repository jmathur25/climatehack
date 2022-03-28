import einops
import torch
import torch.nn.functional as F
from torch.nn.modules.pixelshuffle import PixelShuffle
from torch.nn.utils.parametrizations import spectral_norm
from typing import List
from dgmr.common import GBlock, UpsampleGBlock
from dgmr.layers import ConvGRU
from huggingface_hub import PyTorchModelHubMixin
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN)


class Sampler(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        forecast_steps: int = 18,
        context_channels: int = 384,
        latent_channels: int = 384,
        output_channels: int = 1,
        **kwargs
    ):
        """
        Sampler from the Skillful Nowcasting, see https://arxiv.org/pdf/2104.00954.pdf

        The sampler takes the output from the Latent and Context conditioning stacks and
        creates one stack of ConvGRU layers per future timestep.
        Args:
            forecast_steps: Number of forecast steps
            latent_channels: Number of input channels to the lowest ConvGRU layer
        """
        super().__init__()
        config = locals()
        config.pop("__class__")
        config.pop("self")
        self.config = kwargs.get("config", config)
        self.forecast_steps = self.config["forecast_steps"]
        latent_channels = self.config["latent_channels"]
        context_channels = self.config["context_channels"]
        output_channels = self.config["output_channels"]

        self.gru_conv_1x1 = spectral_norm(
            torch.nn.Conv2d(
                in_channels=context_channels,
                out_channels=latent_channels * self.forecast_steps,
                kernel_size=(1, 1),
            )
        )
        self.g1 = GBlock(
            input_channels=latent_channels * self.forecast_steps,
            output_channels=latent_channels * self.forecast_steps,
        )
        self.up_g1 = UpsampleGBlock(
            input_channels=latent_channels * self.forecast_steps,
            output_channels=latent_channels * self.forecast_steps // 2,
        )

        self.gru_conv_1x1_2 = spectral_norm(
            torch.nn.Conv2d(
                in_channels=self.up_g1.output_channels + context_channels // 2,
                out_channels=latent_channels * self.forecast_steps // 2,
                kernel_size=(1, 1),
            )
        )
        self.g2 = GBlock(
            input_channels=latent_channels * self.forecast_steps // 2,
            output_channels=latent_channels * self.forecast_steps // 2,
        )
        self.up_g2 = UpsampleGBlock(
            input_channels=latent_channels * self.forecast_steps // 2,
            output_channels=latent_channels * self.forecast_steps // 4,
        )

        self.gru_conv_1x1_3 = spectral_norm(
            torch.nn.Conv2d(
                in_channels=self.up_g2.output_channels + context_channels // 4,
                out_channels=latent_channels * self.forecast_steps // 4,
                kernel_size=(1, 1),
            )
        )
        self.g3 = GBlock(
            input_channels=latent_channels * self.forecast_steps // 4,
            output_channels=latent_channels * self.forecast_steps // 4,
        )
        self.up_g3 = UpsampleGBlock(
            input_channels=latent_channels * self.forecast_steps // 4,
            output_channels=latent_channels * self.forecast_steps // 8,
        )

        self.gru_conv_1x1_4 = spectral_norm(
            torch.nn.Conv2d(
                in_channels=self.up_g3.output_channels + context_channels // 8,
                out_channels=latent_channels * self.forecast_steps // 8,
                kernel_size=(1, 1),
            )
        )
        self.g4 = GBlock(
            input_channels=latent_channels * self.forecast_steps // 8,
            output_channels=latent_channels * self.forecast_steps // 8,
        )
        self.up_g4 = UpsampleGBlock(
            input_channels=latent_channels * self.forecast_steps // 8,
            output_channels=latent_channels * self.forecast_steps // 16,
        )

        self.bn = torch.nn.BatchNorm2d(latent_channels * self.forecast_steps // 16)
        self.relu = torch.nn.ReLU()
        self.conv_1x1 = spectral_norm(
            torch.nn.Conv2d(
                in_channels=latent_channels * self.forecast_steps // 16,
                out_channels=4 * output_channels * self.forecast_steps,
                kernel_size=(1, 1),
            )
        )

        self.depth2space = PixelShuffle(upscale_factor=2)

    def forward(self, conditioning_states: List[torch.Tensor]) -> torch.Tensor:
        """
        Perform the sampling from Skillful Nowcasting with GANs
        Args:
            conditioning_states: Outputs from the `ContextConditioningStack` with the 4 input states, ordered from largest to smallest spatially
            latent_dim: Output from `LatentConditioningStack` for input into the ConvGRUs

        Returns:
            forecast_steps-length output of images for future timesteps

        """
        # Iterate through each forecast step
        # Initialize with conditioning state for first one, output for second one
        init_states = conditioning_states

        layer4_states = self.gru_conv_1x1(init_states[3])
        layer4_states = self.g1(layer4_states)
        layer4_states = self.up_g1(layer4_states)

        # Layer 3.
        layer3_states = torch.cat([layer4_states, init_states[2]], dim=1)
        layer3_states = self.gru_conv_1x1_2(layer3_states)
        layer3_states = self.g2(layer3_states)
        layer3_states = self.up_g2(layer3_states)

        # Layer 2.
        layer2_states = torch.cat([layer3_states, init_states[1]], dim=1)
        layer2_states = self.gru_conv_1x1_3(layer2_states)
        layer2_states = self.g3(layer2_states)
        layer2_states = self.up_g3(layer2_states)

        # Layer 1 (top-most).
        layer1_states = torch.cat([layer2_states, init_states[0]], dim=1)
        layer1_states = self.gru_conv_1x1_4(layer1_states)
        layer1_states = self.g4(layer1_states)
        layer1_states = self.up_g4(layer1_states)

        # Final stuff
        output_states = self.relu(self.bn(layer1_states))
        output_states = self.conv_1x1(output_states)
        output_states = self.depth2space(output_states)

        # The satellite dimension was lost, add it back
        output_states = torch.unsqueeze(output_states, dim=2)

        return output_states


class Generator(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        conditioning_stack: torch.nn.Module,
        sampler: torch.nn.Module,
    ):
        """
        Wraps the three parts of the generator for simpler calling
        Args:
            conditioning_stack:
            latent_stack:
            sampler:
        """
        super().__init__()
        self.conditioning_stack = conditioning_stack
        self.sampler = sampler

    def forward(self, x):
        conditioning_states = self.conditioning_stack(x)
        x = self.sampler(conditioning_states)
        return x
