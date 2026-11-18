import torch
import torch.nn as nn
import torch.nn.functional as F


def make_norm3d(channels, norm_type="batch", max_groups=8):
    norm = str(norm_type or "batch").lower()
    if norm in {"batch", "bn", "batchnorm"}:
        return nn.BatchNorm3d(channels)
    if norm in {"group", "gn", "groupnorm"}:
        groups = min(int(max_groups), int(channels))
        while groups > 1 and channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    if norm in {"instance", "in", "instancenorm"}:
        return nn.InstanceNorm3d(channels, affine=True)
    raise ValueError(f"unknown norm_type: {norm_type}")


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1, use_attn=False, reduction=16, norm_type="batch"):
        super().__init__()
        g1 = groups if (in_channels % groups == 0 and out_channels % groups == 0) else 1
        g2 = groups if (out_channels % groups == 0) else 1
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, groups=g1, bias=False)
        self.bn1 = make_norm3d(out_channels, norm_type)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, groups=g2, bias=False)
        self.bn2 = make_norm3d(out_channels, norm_type)
        self.relu2 = nn.ReLU(inplace=True)
        self.attn = SEBlock(out_channels, reduction=reduction) if use_attn else nn.Identity()

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return self.attn(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1, use_attn=False, reduction=16, norm_type="batch"):
        super().__init__()
        g1 = groups if (in_channels % groups == 0 and out_channels % groups == 0) else 1
        g2 = groups if (out_channels % groups == 0) else 1
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, groups=g1, bias=False)
        self.bn1 = make_norm3d(out_channels, norm_type)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, groups=g2, bias=False)
        self.bn2 = make_norm3d(out_channels, norm_type)
        self.relu2 = nn.ReLU(inplace=True)
        if in_channels == out_channels:
            self.skip_proj = nn.Identity()
        else:
            self.skip_proj = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                make_norm3d(out_channels, norm_type),
            )
        # Start from the pretrained non-residual behavior and learn residual mixing during adaptation.
        self.res_scale = nn.Parameter(torch.zeros(1))
        self.attn = SEBlock(out_channels, reduction=reduction) if use_attn else nn.Identity()

    def forward(self, x):
        skip = self.skip_proj(x)
        y = self.relu1(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y = self.relu2(y + self.res_scale * skip)
        return self.attn(y)


class PreActResidualBlock(nn.Module):
    """Batch-size-one friendly residual block for full retraining."""

    def __init__(self, in_channels, out_channels, groups=1, use_attn=False, reduction=16, norm_type="group"):
        super().__init__()
        g1 = groups if (in_channels % groups == 0 and out_channels % groups == 0) else 1
        g2 = groups if out_channels % groups == 0 else 1
        self.norm1 = make_norm3d(in_channels, norm_type)
        self.act1 = nn.SiLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, groups=g1, bias=False)
        self.norm2 = make_norm3d(out_channels, norm_type)
        self.act2 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, groups=g2, bias=False)
        self.skip_proj = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.attn = SEBlock(out_channels, reduction=reduction) if use_attn else nn.Identity()

    def forward(self, x):
        skip = self.skip_proj(x)
        y = self.conv1(self.act1(self.norm1(x)))
        y = self.conv2(self.act2(self.norm2(y)))
        return self.attn(skip + y)


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        skip_channels=0,
        groups=1,
        use_attn=False,
        reduction=16,
        block_cls=ConvBlock,
        norm_type="batch",
    ):
        super().__init__()
        g = groups if (in_channels % groups == 0 and out_channels % groups == 0) else 1
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2, groups=g, bias=False)
        self.bn = make_norm3d(out_channels, norm_type)
        self.relu = nn.ReLU(inplace=True)
        self.skip_channels = skip_channels
        self.conv_block = block_cls(
            out_channels + (skip_channels if skip_channels else 0),
            out_channels,
            groups=groups,
            use_attn=use_attn,
            reduction=reduction,
            norm_type=norm_type,
        )

    def forward(self, x, skip=None):
        x = self.relu(self.bn(self.up(x)))
        if self.skip_channels and skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv_block(x)


class ASPP3DBlock(nn.Module):
    def __init__(self, channels, groups=1, use_attn=False, reduction=16, block_cls=ConvBlock, norm_type="batch"):
        super().__init__()
        mid = max(32, channels // 4)

        def maybe_groups(cin, cout):
            return groups if (cin % groups == 0 and cout % groups == 0) else 1

        self.b1 = nn.Sequential(
            nn.Conv3d(channels, mid, kernel_size=1, bias=False),
            make_norm3d(mid, norm_type),
            nn.ReLU(inplace=True),
        )
        self.b2 = nn.Sequential(
            nn.Conv3d(channels, mid, kernel_size=3, padding=1, dilation=1, groups=maybe_groups(channels, mid), bias=False),
            make_norm3d(mid, norm_type),
            nn.ReLU(inplace=True),
        )
        self.b3 = nn.Sequential(
            nn.Conv3d(channels, mid, kernel_size=3, padding=2, dilation=2, groups=maybe_groups(channels, mid), bias=False),
            make_norm3d(mid, norm_type),
            nn.ReLU(inplace=True),
        )
        self.b4 = nn.Sequential(
            nn.Conv3d(channels, mid, kernel_size=3, padding=4, dilation=4, groups=maybe_groups(channels, mid), bias=False),
            make_norm3d(mid, norm_type),
            nn.ReLU(inplace=True),
        )
        self.fuse = block_cls(
            mid * 4,
            channels,
            groups=groups,
            use_attn=use_attn,
            reduction=reduction,
            norm_type=norm_type,
        )

    def forward(self, x):
        y = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
        return self.fuse(y)


class RouteContextModulation(nn.Module):
    """Identity-initialized modulation from route confidence and observations."""

    def __init__(self, channels):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))
        self.context_adjust = nn.Conv3d(2, 1, kernel_size=3, padding=1)
        nn.init.zeros_(self.context_adjust.weight)
        nn.init.zeros_(self.context_adjust.bias)

    def forward(self, x, context):
        if context.shape[2:] != x.shape[2:]:
            context = F.interpolate(context, size=x.shape[2:], mode="trilinear", align_corners=False)
        confidence = context[:, :1].clamp(0.0, 1.0)
        observed = context[:, 1:2].clamp(0.0, 1.0)
        learned = torch.tanh(self.context_adjust(context))
        field = torch.clamp(2.0 * confidence - 1.0 + 0.5 * observed + learned, -1.5, 1.5)
        return x * (1.0 + torch.tanh(self.scale) * field) + self.bias * field


class RouteConfidenceResidualHead(nn.Module):
    """Near-route/local and far-route/context residual experts."""

    def __init__(self, channels, out_channels):
        super().__init__()
        self.local_head = nn.Conv3d(channels, out_channels, kernel_size=3, padding=1)
        self.context_head = nn.Conv3d(channels, out_channels, kernel_size=3, padding=3, dilation=3)
        self.gate_adjust = nn.Conv3d(2, 1, kernel_size=3, padding=1)
        for module in (self.local_head, self.context_head, self.gate_adjust):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, features, context):
        if context.shape[2:] != features.shape[2:]:
            context = F.interpolate(context, size=features.shape[2:], mode="trilinear", align_corners=False)
        confidence = context[:, :1].clamp(0.0, 1.0)
        gate = torch.clamp(confidence + 0.25 * torch.tanh(self.gate_adjust(context)), 0.0, 1.0)
        local_residual = self.local_head(features)
        context_residual = self.context_head(features)
        return gate * local_residual + (1.0 - gate) * context_residual


class RouteAnisotropicResidualHead(nn.Module):
    """Propagate context along the weakly constrained route-normal direction."""

    def __init__(self, channels, out_channels, use_route_gradient=True):
        super().__init__()
        self.use_route_gradient = bool(use_route_gradient)
        self.local_head = nn.Conv3d(channels, out_channels, kernel_size=3, padding=1)
        self.axis_heads = nn.ModuleList(
            [
                nn.Conv3d(channels, out_channels, kernel_size=(7, 1, 1), padding=(3, 0, 0)),
                nn.Conv3d(channels, out_channels, kernel_size=(1, 7, 1), padding=(0, 3, 0)),
                nn.Conv3d(channels, out_channels, kernel_size=(1, 1, 7), padding=(0, 0, 3)),
            ]
        )
        self.direction_bias = nn.Parameter(torch.zeros(1, 3, 1, 1, 1))
        nn.init.zeros_(self.local_head.weight)
        nn.init.zeros_(self.local_head.bias)
        for head in self.axis_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    @staticmethod
    def _axis_gradient(confidence, axis):
        if axis == 2:
            diff = torch.abs(confidence[:, :, 1:] - confidence[:, :, :-1])
            return F.pad(diff, (0, 0, 0, 0, 0, 1))
        if axis == 3:
            diff = torch.abs(confidence[:, :, :, 1:] - confidence[:, :, :, :-1])
            return F.pad(diff, (0, 0, 0, 1, 0, 0))
        diff = torch.abs(confidence[:, :, :, :, 1:] - confidence[:, :, :, :, :-1])
        return F.pad(diff, (0, 1, 0, 0, 0, 0))

    def forward(self, features, context):
        if context.shape[2:] != features.shape[2:]:
            context = F.interpolate(context, size=features.shape[2:], mode="trilinear", align_corners=False)
        confidence = context[:, :1].clamp(0.0, 1.0)
        if self.use_route_gradient:
            gradients = torch.cat(
                [self._axis_gradient(confidence, axis) for axis in (2, 3, 4)],
                dim=1,
            )
            direction_logits = torch.log(gradients + 1e-4) + self.direction_bias
        else:
            direction_logits = self.direction_bias.expand(
                confidence.shape[0], -1, *confidence.shape[2:]
            )
        direction_weights = torch.softmax(direction_logits, dim=1)
        directional = sum(
            direction_weights[:, axis : axis + 1] * head(features)
            for axis, head in enumerate(self.axis_heads)
        )
        local = self.local_head(features)
        return confidence * local + (1.0 - confidence) * directional


class RouteHybridResidualHead(nn.Module):
    """Blend isotropic and route-normal context using directional confidence."""

    def __init__(self, channels, out_channels):
        super().__init__()
        self.local_head = nn.Conv3d(channels, out_channels, kernel_size=3, padding=1)
        self.isotropic_head = nn.Conv3d(
            channels,
            out_channels,
            kernel_size=3,
            padding=3,
            dilation=3,
        )
        self.axis_heads = nn.ModuleList(
            [
                nn.Conv3d(channels, out_channels, kernel_size=(7, 1, 1), padding=(3, 0, 0)),
                nn.Conv3d(channels, out_channels, kernel_size=(1, 7, 1), padding=(0, 3, 0)),
                nn.Conv3d(channels, out_channels, kernel_size=(1, 1, 7), padding=(0, 0, 3)),
            ]
        )
        self.direction_bias = nn.Parameter(torch.zeros(1, 3, 1, 1, 1))
        nn.init.zeros_(self.local_head.weight)
        nn.init.zeros_(self.local_head.bias)
        nn.init.zeros_(self.isotropic_head.weight)
        nn.init.zeros_(self.isotropic_head.bias)
        for head in self.axis_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, features, context):
        if context.shape[2:] != features.shape[2:]:
            context = F.interpolate(context, size=features.shape[2:], mode="trilinear", align_corners=False)
        confidence = context[:, :1].clamp(0.0, 1.0)
        gradients = torch.cat(
            [RouteAnisotropicResidualHead._axis_gradient(confidence, axis) for axis in (2, 3, 4)],
            dim=1,
        )
        direction_weights = torch.softmax(torch.log(gradients + 1e-4) + self.direction_bias, dim=1)
        directional = sum(
            direction_weights[:, axis : axis + 1] * head(features)
            for axis, head in enumerate(self.axis_heads)
        )
        gradient_sum = gradients.sum(dim=1, keepdim=True)
        dominance = torch.clamp(
            0.5 * (3.0 * gradients.max(dim=1, keepdim=True).values / (gradient_sum + 1e-4) - 1.0),
            0.0,
            1.0,
        )
        gradient_presence = 1.0 - torch.exp(-20.0 * gradient_sum)
        anisotropic_gate = dominance * gradient_presence
        far_context = (
            anisotropic_gate * directional
            + (1.0 - anisotropic_gate) * self.isotropic_head(features)
        )
        return confidence * self.local_head(features) + (1.0 - confidence) * far_context


class RouteMultiScaleHybridResidualHead(nn.Module):
    """Fuse full- and half-resolution route-normal corrections."""

    def __init__(self, full_channels, half_channels, out_channels):
        super().__init__()
        self.full_head = RouteHybridResidualHead(full_channels, out_channels)
        self.half_head = RouteHybridResidualHead(half_channels, out_channels)
        self.half_scale = nn.Parameter(torch.ones(1))

    def forward(self, full_features, half_features, context):
        full_residual = self.full_head(full_features, context)
        half_residual = self.half_head(half_features, context)
        half_residual = F.interpolate(
            half_residual,
            size=full_residual.shape[2:],
            mode="trilinear",
            align_corners=False,
        )
        return full_residual + self.half_scale * half_residual


class UNet3DCoarseFine(nn.Module):
    def __init__(
        self,
        in_channels=1,
        base_channels=32,
        out_channels=1,
        use_wavelet=True,
        use_attn=False,
        group_conv=False,
        bottleneck_mode=None,
        append_obs_mask=False,
        conv_block_type="standard",
        norm_type="batch",
        prior_fusion_mode="first",
        prior_channels=1,
        route_context_mode="none",
        route_confidence_channel=3,
        route_observation_channel=4,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.append_obs_mask = bool(append_obs_mask)
        self.use_attn = bool(use_attn)
        self.groups = 8 if group_conv else 1
        self.bottleneck_mode = str(bottleneck_mode or ("wavelet" if use_wavelet else "plain")).lower()
        self.use_wavelet = self.bottleneck_mode == "wavelet"
        self.conv_block_type = str(conv_block_type or "standard").lower()
        self.norm_type = str(norm_type or "batch").lower()
        self.prior_fusion_mode = str(prior_fusion_mode or "first").lower()
        self.prior_channels = max(1, min(int(prior_channels), self.in_channels))
        self.route_context_mode = str(route_context_mode or "none").lower()
        self.route_confidence_channel = int(route_confidence_channel)
        self.route_observation_channel = int(route_observation_channel)
        valid_route_modes = {
            "none",
            "modulate",
            "dual_head",
            "modulate_dual",
            "anisotropic_head",
            "modulate_anisotropic",
            "axis_head",
            "hybrid_head",
            "multiscale_hybrid_head",
        }
        if self.route_context_mode not in valid_route_modes:
            raise ValueError(f"unknown route_context_mode: {route_context_mode}")
        if self.route_context_mode != "none":
            max_context_channel = max(self.route_confidence_channel, self.route_observation_channel)
            if min(self.route_confidence_channel, self.route_observation_channel) < 0 or max_context_channel >= self.in_channels:
                raise ValueError(
                    "route context channels must index available model inputs: "
                    f"confidence={self.route_confidence_channel}, observation={self.route_observation_channel}, "
                    f"in_channels={self.in_channels}"
                )
        if self.conv_block_type in ("standard", "conv", "base"):
            block_cls = ConvBlock
        elif self.conv_block_type in ("residual", "res", "resblock"):
            block_cls = ResidualConvBlock
            self.conv_block_type = "residual"
        elif self.conv_block_type in ("residual_v2", "resv2", "preact_residual"):
            block_cls = PreActResidualBlock
            self.conv_block_type = "residual_v2"
        else:
            raise ValueError(f"unknown conv_block_type: {self.conv_block_type}")

        if self.prior_fusion_mode in {"learned", "learned_softmax", "softmax"}:
            hidden = max(4, min(16, self.in_channels * 2))
            self.prior_gate = nn.Sequential(
                nn.Conv3d(self.in_channels, hidden, kernel_size=1),
                nn.SiLU(inplace=True),
                nn.Conv3d(hidden, self.prior_channels, kernel_size=1),
            )
            nn.init.zeros_(self.prior_gate[-1].weight)
            nn.init.zeros_(self.prior_gate[-1].bias)
            if self.prior_channels > 1:
                with torch.no_grad():
                    self.prior_gate[-1].bias[0] = 2.0
                    self.prior_gate[-1].bias[1:] = -2.0
            self.prior_fusion_mode = "learned_softmax"
        elif self.prior_fusion_mode in {"first", "none", "fixed"}:
            self.prior_gate = None
            self.prior_fusion_mode = "first"
        else:
            raise ValueError(f"unknown prior_fusion_mode: {prior_fusion_mode}")

        block_kwargs = dict(groups=self.groups, use_attn=self.use_attn, norm_type=self.norm_type)
        self.enc0 = block_cls(self.in_channels, base_channels, **block_kwargs)
        self.down1 = nn.MaxPool3d(2)
        self.enc1 = block_cls(base_channels, base_channels * 2, **block_kwargs)
        self.down2 = nn.MaxPool3d(2)
        self.enc2 = block_cls(base_channels * 2, base_channels * 4, **block_kwargs)
        self.down3 = nn.MaxPool3d(2)
        self.enc3 = block_cls(base_channels * 4, base_channels * 8, **block_kwargs)
        self.down4 = nn.MaxPool3d(2)
        self.enc4 = block_cls(base_channels * 8, base_channels * 16, **block_kwargs)

        bottleneck_channels = base_channels * 16
        if self.bottleneck_mode == "wavelet":
            self.low_conv = block_cls(bottleneck_channels, bottleneck_channels, **block_kwargs)
            self.high_conv = nn.Conv3d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1, bias=False)
            self.high_bn = make_norm3d(bottleneck_channels, self.norm_type)
            self.high_relu = nn.ReLU(inplace=True)
        elif self.bottleneck_mode == "plain":
            self.bottleneck_conv = block_cls(bottleneck_channels, bottleneck_channels, **block_kwargs)
        elif self.bottleneck_mode == "aspp":
            self.bottleneck_conv = ASPP3DBlock(
                bottleneck_channels,
                groups=self.groups,
                use_attn=self.use_attn,
                block_cls=block_cls,
                norm_type=self.norm_type,
            )
        else:
            raise ValueError(f"unknown bottleneck_mode: {self.bottleneck_mode}")

        self.coarse_up1 = UpBlock(bottleneck_channels, base_channels * 8, skip_channels=0, groups=self.groups, use_attn=self.use_attn, block_cls=block_cls, norm_type=self.norm_type)
        self.coarse_up2 = UpBlock(base_channels * 8, base_channels * 4, skip_channels=0, groups=self.groups, use_attn=self.use_attn, block_cls=block_cls, norm_type=self.norm_type)
        self.coarse_up3 = UpBlock(base_channels * 4, base_channels * 2, skip_channels=0, groups=self.groups, use_attn=self.use_attn, block_cls=block_cls, norm_type=self.norm_type)
        self.coarse_out_conv = nn.Conv3d(base_channels * 2, out_channels, kernel_size=1)

        self.up1 = UpBlock(bottleneck_channels, base_channels * 8, skip_channels=base_channels * 8, groups=self.groups, use_attn=self.use_attn, block_cls=block_cls, norm_type=self.norm_type)
        self.up2 = UpBlock(base_channels * 8, base_channels * 4, skip_channels=base_channels * 4, groups=self.groups, use_attn=self.use_attn, block_cls=block_cls, norm_type=self.norm_type)
        self.up3 = UpBlock(base_channels * 4, base_channels * 2, skip_channels=base_channels * 2, groups=self.groups, use_attn=self.use_attn, block_cls=block_cls, norm_type=self.norm_type)
        self.up4 = UpBlock(base_channels * 2, base_channels, skip_channels=base_channels, groups=self.groups, use_attn=self.use_attn, block_cls=block_cls, norm_type=self.norm_type)
        self.fine_out_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)
        self.final_fuse_conv = nn.Conv3d(out_channels * 2, out_channels, kernel_size=1)

        if self.route_context_mode in {"modulate", "modulate_dual", "modulate_anisotropic"}:
            modulation_channels = {
                "enc0": base_channels,
                "enc1": base_channels * 2,
                "enc2": base_channels * 4,
                "enc3": base_channels * 8,
                "enc4": base_channels * 16,
                "dec1": base_channels * 8,
                "dec2": base_channels * 4,
                "dec3": base_channels * 2,
                "dec4": base_channels,
            }
            self.route_modulation = nn.ModuleDict(
                {name: RouteContextModulation(channels) for name, channels in modulation_channels.items()}
            )
        else:
            self.route_modulation = None

        if self.route_context_mode in {"dual_head", "modulate_dual"}:
            self.route_residual_head = RouteConfidenceResidualHead(base_channels, out_channels)
        elif self.route_context_mode in {"anisotropic_head", "modulate_anisotropic"}:
            self.route_residual_head = RouteAnisotropicResidualHead(base_channels, out_channels)
        elif self.route_context_mode == "axis_head":
            self.route_residual_head = RouteAnisotropicResidualHead(
                base_channels,
                out_channels,
                use_route_gradient=False,
            )
        elif self.route_context_mode == "hybrid_head":
            self.route_residual_head = RouteHybridResidualHead(base_channels, out_channels)
        elif self.route_context_mode == "multiscale_hybrid_head":
            self.route_residual_head = RouteMultiScaleHybridResidualHead(
                base_channels,
                base_channels * 2,
                out_channels,
            )
        else:
            self.route_residual_head = None

    def forward(self, x, mask=None):
        if mask is None:
            mask = (x[:, :1] == 0).float()
        if self.prior_gate is None:
            base_signal = x[:, :1]
        else:
            prior_weights = torch.softmax(self.prior_gate(x), dim=1)
            base_signal = (prior_weights * x[:, : self.prior_channels]).sum(dim=1, keepdim=True)

        route_context = None
        if self.route_context_mode != "none":
            route_context = torch.cat(
                [
                    x[:, self.route_confidence_channel : self.route_confidence_channel + 1],
                    x[:, self.route_observation_channel : self.route_observation_channel + 1],
                ],
                dim=1,
            )

        e0 = self.enc0(x)
        if self.route_modulation is not None:
            e0 = self.route_modulation["enc0"](e0, route_context)
        e1 = self.enc1(self.down1(e0))
        if self.route_modulation is not None:
            e1 = self.route_modulation["enc1"](e1, route_context)
        e2 = self.enc2(self.down2(e1))
        if self.route_modulation is not None:
            e2 = self.route_modulation["enc2"](e2, route_context)
        e3 = self.enc3(self.down3(e2))
        if self.route_modulation is not None:
            e3 = self.route_modulation["enc3"](e3, route_context)
        e4 = self.enc4(self.down4(e3))
        if self.route_modulation is not None:
            e4 = self.route_modulation["enc4"](e4, route_context)

        if self.bottleneck_mode == "wavelet":
            low = F.avg_pool3d(e4, kernel_size=2, stride=2)
            low_proc = self.low_conv(low)
            low_up = F.interpolate(low_proc, size=e4.shape[2:], mode="trilinear", align_corners=False)
            high = e4 - low_up
            high = self.high_relu(self.high_bn(self.high_conv(high)))
            bottleneck_out = low_up + high
        else:
            bottleneck_out = self.bottleneck_conv(e4)

        c = self.coarse_up1(bottleneck_out)
        c = self.coarse_up2(c)
        c = self.coarse_up3(c)
        coarse_out = self.coarse_out_conv(c)

        f = self.up1(bottleneck_out, skip=e3)
        if self.route_modulation is not None:
            f = self.route_modulation["dec1"](f, route_context)
        f = self.up2(f, skip=e2)
        if self.route_modulation is not None:
            f = self.route_modulation["dec2"](f, route_context)
        f = self.up3(f, skip=e1)
        if self.route_modulation is not None:
            f = self.route_modulation["dec3"](f, route_context)
        half_f = f
        f = self.up4(f, skip=e0)
        if self.route_modulation is not None:
            f = self.route_modulation["dec4"](f, route_context)
        fine_out = self.fine_out_conv(f)

        coarse_up = F.interpolate(coarse_out, size=fine_out.shape[2:], mode="trilinear", align_corners=False)
        fused = torch.cat([fine_out, coarse_up], dim=1)
        residual = self.final_fuse_conv(fused)
        if self.route_residual_head is not None:
            if self.route_context_mode == "multiscale_hybrid_head":
                residual = residual + self.route_residual_head(f, half_f, route_context)
            else:
                residual = residual + self.route_residual_head(f, route_context)
        return base_signal + residual * mask
