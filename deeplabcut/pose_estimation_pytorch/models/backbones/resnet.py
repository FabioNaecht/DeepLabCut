#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/main/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
import timm
from timm import create_model
import torch
import torch.nn as nn
from torchvision.transforms.functional import resize

from deeplabcut.pose_estimation_pytorch.models.backbones.base import (
    BACKBONES,
    BaseBackbone,
)


@BACKBONES.register_module
class ResNet(BaseBackbone):
    """MobileNet backbone"""

    def __init__(
        self,
        model_name: str = "mobilenetv3_small_100",
        output_stride: int = 32,
        pretrained: bool = False,
        input_channels: int = 3,  # Allow grayscale input
        expected_output_channels: int = 2048,
        **kwargs,
    ) -> None:
        """Initialize the MobileNet backbone"""
        print(f"################# model name: {model_name} #################")
        super().__init__(stride=output_stride, **kwargs)
        self.model = create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,  # Enable feature extraction
            out_indices=(-1,),  # Extract only the last feature map
        )
        if input_channels == 1:
            self.model.conv_stem = nn.Conv2d(
                in_channels=1,
                out_channels=self.model.conv_stem.out_channels,
                kernel_size=self.model.conv_stem.kernel_size,
                stride=self.model.conv_stem.stride,
                padding=self.model.conv_stem.padding,
                bias=False,
            )
        self.output_conv = nn.Conv2d(
            in_channels=self.model.feature_info[-1]['num_chs'],
            out_channels=expected_output_channels,  # Match head input
            kernel_size=1,
            stride=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model(x)  # features_only=True returns a list of feature maps
        last_feature_map = features[-1]  # Extract the last feature map
        print(f"Last feature map shape: {last_feature_map.shape}")  # Debugging
        out = self.output_conv(last_feature_map)  # Adjust channels to expected output
        return out

@BACKBONES.register_module
class ResNet1(BaseBackbone):
    """ResNet backbone.

    This class represents a typical ResNet backbone for pose estimation.

    Attributes:
        model: the ResNet model
    """

    def __init__(
        self,
        model_name: str = "resnet50",
        output_stride: int = 32,
        pretrained: bool = False,
        drop_path_rate: float = 0.0,
        drop_block_rate: float = 0.0,
        **kwargs,
    ) -> None:
        """Initialize the ResNet backbone.

        Args:
            model_name: Name of the ResNet model to use, e.g., 'resnet50', 'resnet101'
            output_stride: Output stride of the network, 32, 16, or 8.
            pretrained: If True, initializes with ImageNet pretrained weights.
            drop_path_rate: Stochastic depth drop-path rate
            drop_block_rate: Drop block rate
            kwargs: BaseBackbone kwargs
        """
        super().__init__(stride=output_stride, **kwargs)
        self.model = timm.create_model(
            model_name,
            output_stride=output_stride,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
            drop_block_rate=drop_block_rate,
        )
        self.model.fc = nn.Identity()  # remove the FC layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet backbone.

        Args:
            x: Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        Example:
            >>> import torch
            >>> from deeplabcut.pose_estimation_pytorch.models.backbones import ResNet
            >>> backbone = ResNet(model_name='resnet50', pretrained=False)
            >>> x = torch.randn(1, 3, 256, 256)
            >>> y = backbone(x)

            Expected Output Shape:
                If input size is (batch_size, 3, shape_x, shape_y), the output shape
                will be (batch_size, 3, shape_x//16, shape_y//16)
        """
        return self.model.forward_features(x)

@BACKBONES.register_module
class DLCRNet(ResNet):
    def __init__(
        self,
        model_name: str = "resnet50",
        # model_name: str = "mobilenetv3_small_100",
        output_stride: int = 32,
        pretrained: bool = True,
        **kwargs,
    ) -> None:
        print(f"----------- model name: {model_name} ---------------")
        super().__init__(model_name, output_stride, pretrained, **kwargs)
        self.interm_features = {}
        self.model.layer1[2].register_forward_hook(self._get_features("bank1"))
        self.model.layer2[2].register_forward_hook(self._get_features("bank2"))
        self.conv_block1 = self._make_conv_block(
            in_channels=512, out_channels=512, kernel_size=3, stride=2
        )
        self.conv_block2 = self._make_conv_block(
            in_channels=512, out_channels=128, kernel_size=1, stride=1
        )
        self.conv_block3 = self._make_conv_block(
            in_channels=256, out_channels=256, kernel_size=3, stride=2
        )
        self.conv_block4 = self._make_conv_block(
            in_channels=256, out_channels=256, kernel_size=3, stride=2
        )
        self.conv_block5 = self._make_conv_block(
            in_channels=256, out_channels=128, kernel_size=1, stride=1
        )

    def _make_conv_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        momentum: float = 0.001,  # (1 - decay)
    ) -> torch.nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, stride=stride
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )

    def _get_features(self, name):
        def hook(model, input, output):
            self.interm_features[name] = output.detach()

        return hook

    def forward(self, x):
        out = super().forward(x)

        # Fuse intermediate features
        bank_2_s8 = self.interm_features["bank2"]
        bank_1_s4 = self.interm_features["bank1"]
        bank_2_s16 = self.conv_block1(bank_2_s8)
        bank_2_s16 = self.conv_block2(bank_2_s16)
        bank_1_s8 = self.conv_block3(bank_1_s4)
        bank_1_s16 = self.conv_block4(bank_1_s8)
        bank_1_s16 = self.conv_block5(bank_1_s16)
        # Resizing here is required to guarantee all shapes match, as
        # Conv2D(..., padding='same') is invalid for strided convolutions.
        h, w = out.shape[-2:]
        bank_1_s16 = resize(bank_1_s16, [h, w], antialias=True)
        bank_2_s16 = resize(bank_2_s16, [h, w], antialias=True)

        return torch.cat((bank_1_s16, bank_2_s16, out), dim=1)

@BACKBONES.register_module
class MobileNetV3Small(BaseBackbone):
    """MobileNetV3Small backbone.

    Attributes:
        model: the MobileNetV3Small model.
    """

    def __init__(
            self,
            model_name: str = "mobilenetv3_small_100",
            output_stride: int = 16,
            pretrained: bool = True,
            drop_path_rate: bool = True,
            drop_block_rate: bool = True,
            **kwargs,
    ) -> None:
        """Initialize the MobileNetV3Small backbone.

        Args:
            model_name: Name of the MobileNet model to use.
            output_stride: Output stride of the network, typically 32.
            pretrained: If True, initializes with ImageNet pretrained weights.
            kwargs: BaseBackbone kwargs.
        """
        super().__init__(stride=output_stride, **kwargs)
        self.model = timm.create_model(
            model_name, pretrained=pretrained, drop_path_rate=drop_path_rate, drop_block_rate=drop_block_rate, features_only=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MobileNetV3Small backbone."""
        return self.model(x)

