from __future__ import annotations

import math

import torch
import torch.nn as nn


TIME_YAW_TVI_MODE = "time_yaw"
TIME_CAMERA_POSE_TVI_MODE = "time_camera_pose"
METRIC_CAMERA_POSE_TVI_MODE = "metric_camera_pose"
LEARNED_TOKEN_TVI_MODE = "learned_token"
CAMERA_POSE_TVI_MODES = frozenset({TIME_CAMERA_POSE_TVI_MODE, METRIC_CAMERA_POSE_TVI_MODE})

METRIC_TIME_SCALE_SECONDS = 4096.0
METRIC_TIME_LOG_SCALE_SECONDS = 10.0
METRIC_TIME_MIN_WAVELENGTH_SECONDS = 0.2
METRIC_TIME_MAX_WAVELENGTH_SECONDS = 4096.0
METRIC_POSITION_SCALE_METERS = 2048.0
METRIC_POSITION_LOG_SCALE_METERS = 16.0
METRIC_POSITION_MIN_WAVELENGTH_METERS = 1.0
METRIC_POSITION_MAX_WAVELENGTH_METERS = 2048.0
METRIC_FOURIER_WAVELENGTHS = 16
METRIC_TIME_FEATURE_DIM = 2 + 2 * METRIC_FOURIER_WAVELENGTHS
METRIC_POSITION_FEATURE_DIM = 6 + 3 * 2 * METRIC_FOURIER_WAVELENGTHS
METRIC_ROTATION_FEATURE_DIM = 6


def get_tvi_input_dim(mode: str) -> int:
    if mode in {TIME_YAW_TVI_MODE, LEARNED_TOKEN_TVI_MODE}:
        return 2
    if mode in CAMERA_POSE_TVI_MODES:
        return 7
    raise ValueError(
        f"unsupported TVI mode {mode!r}; expected one of "
        f"{sorted({TIME_YAW_TVI_MODE, LEARNED_TOKEN_TVI_MODE, *CAMERA_POSE_TVI_MODES})}"
    )


def uses_camera_pose_tvi(mode: str) -> bool:
    return mode in CAMERA_POSE_TVI_MODES


def _geometric_wavelengths(
    minimum: float,
    maximum: float,
    *,
    device: torch.device,
) -> torch.Tensor:
    return torch.logspace(
        math.log10(float(minimum)),
        math.log10(float(maximum)),
        METRIC_FOURIER_WAVELENGTHS,
        device=device,
        dtype=torch.float32,
    )


def _metric_fourier_features(values: torch.Tensor, wavelengths: torch.Tensor) -> torch.Tensor:
    angles = 2.0 * torch.pi * values.unsqueeze(-1) / wavelengths
    return torch.stack([torch.sin(angles), torch.cos(angles)], dim=-1).flatten(start_dim=1)


def metric_camera_pose_features(
    tvi: torch.Tensor,
    *,
    time_wavelengths: torch.Tensor | None = None,
    position_wavelengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if tvi.ndim != 2 or tvi.shape[-1] != 7:
        raise ValueError(f"metric camera-pose TVI tensor must have shape [N, 7], got {tuple(tvi.shape)}")
    values = tvi.to(dtype=torch.float32)
    if time_wavelengths is None:
        time_wavelengths = _geometric_wavelengths(
            METRIC_TIME_MIN_WAVELENGTH_SECONDS,
            METRIC_TIME_MAX_WAVELENGTH_SECONDS,
            device=values.device,
        )
    else:
        time_wavelengths = time_wavelengths.to(device=values.device, dtype=torch.float32)
    if position_wavelengths is None:
        position_wavelengths = _geometric_wavelengths(
            METRIC_POSITION_MIN_WAVELENGTH_METERS,
            METRIC_POSITION_MAX_WAVELENGTH_METERS,
            device=values.device,
        )
    else:
        position_wavelengths = position_wavelengths.to(device=values.device, dtype=torch.float32)

    time = values[:, 0]
    xyz = values[:, 1:4]
    yaw_roll_pitch = values[:, 4:7]
    time_features = torch.cat(
        [
            (time / METRIC_TIME_SCALE_SECONDS).unsqueeze(-1),
            torch.log1p(time / METRIC_TIME_LOG_SCALE_SECONDS).unsqueeze(-1),
            _metric_fourier_features(time.unsqueeze(-1), time_wavelengths),
        ],
        dim=-1,
    )
    position_features = torch.cat(
        [
            xyz / METRIC_POSITION_SCALE_METERS,
            torch.sign(xyz) * torch.log1p(torch.abs(xyz) / METRIC_POSITION_LOG_SCALE_METERS),
            _metric_fourier_features(xyz, position_wavelengths),
        ],
        dim=-1,
    )
    rotation_features = torch.stack(
        [
            torch.sin(yaw_roll_pitch[:, 0]),
            torch.cos(yaw_roll_pitch[:, 0]),
            torch.sin(yaw_roll_pitch[:, 1]),
            torch.cos(yaw_roll_pitch[:, 1]),
            torch.sin(yaw_roll_pitch[:, 2]),
            torch.cos(yaw_roll_pitch[:, 2]),
        ],
        dim=-1,
    )
    return time_features, position_features, rotation_features


def sinusoidal_scalar_pe(values: torch.Tensor, dim: int, *, base: float = 10000.0) -> torch.Tensor:
    if dim <= 0 or dim % 2 != 0:
        raise ValueError(f"sinusoidal PE dim must be a positive even integer, got {dim}")
    values = values.reshape(-1, 1)
    half_dim = dim // 2
    exponent = torch.arange(half_dim, device=values.device, dtype=values.dtype) * (2.0 / float(dim))
    div_term = torch.pow(values.new_tensor(float(base)), exponent)
    angles = values / div_term.unsqueeze(0)
    encoded = torch.empty(values.shape[0], dim, device=values.device, dtype=values.dtype)
    encoded[:, 0::2] = torch.sin(angles)
    encoded[:, 1::2] = torch.cos(angles)
    return encoded


class NavVLATVIEmbedding(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        enable_mask_token: bool = False,
        mode: str = TIME_YAW_TVI_MODE,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        if self.hidden_size <= 0 or self.hidden_size % 4 != 0:
            raise ValueError(f"hidden_size must be a positive integer divisible by 4, got {hidden_size}")
        self.mode = mode
        self.input_dim = get_tvi_input_dim(mode)
        self.base = nn.Parameter(torch.zeros(self.hidden_size))
        if enable_mask_token and self.mode != LEARNED_TOKEN_TVI_MODE:
            self.mask_token = nn.Parameter(torch.zeros(self.hidden_size))
        if self.mode == METRIC_CAMERA_POSE_TVI_MODE:
            self.metric_time_mlp = nn.Sequential(
                nn.Linear(METRIC_TIME_FEATURE_DIM, self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, self.hidden_size),
            )
            self.metric_position_mlp = nn.Sequential(
                nn.Linear(METRIC_POSITION_FEATURE_DIM, self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, self.hidden_size),
            )
            self.metric_rotation_mlp = nn.Sequential(
                nn.Linear(METRIC_ROTATION_FEATURE_DIM, self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, self.hidden_size),
            )
            self.metric_time_gate = nn.Parameter(torch.tensor(1.0))
            self.metric_position_gate = nn.Parameter(torch.tensor(1.0))
            self.metric_rotation_gate = nn.Parameter(torch.tensor(1.0))
            self.register_buffer(
                "metric_time_wavelengths",
                _geometric_wavelengths(
                    METRIC_TIME_MIN_WAVELENGTH_SECONDS,
                    METRIC_TIME_MAX_WAVELENGTH_SECONDS,
                    device=torch.device("cpu"),
                ),
                persistent=False,
            )
            self.register_buffer(
                "metric_position_wavelengths",
                _geometric_wavelengths(
                    METRIC_POSITION_MIN_WAVELENGTH_METERS,
                    METRIC_POSITION_MAX_WAVELENGTH_METERS,
                    device=torch.device("cpu"),
                ),
                persistent=False,
            )
        elif self.mode != LEARNED_TOKEN_TVI_MODE:
            self.time_mlp = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, self.hidden_size),
            )
            self.angle_mlp = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, self.hidden_size),
            )
            if self.mode == TIME_CAMERA_POSE_TVI_MODE:
                self.pose_mlp = nn.Sequential(
                    nn.Linear(196, self.hidden_size),
                    nn.GELU(),
                    nn.Linear(self.hidden_size, self.hidden_size),
                )

    def forward(self, tvi: torch.Tensor) -> torch.Tensor:
        if tvi.ndim != 2 or tvi.shape[-1] != self.input_dim:
            raise ValueError(f"TVI tensor must have shape [N, {self.input_dim}], got {tuple(tvi.shape)}")
        if self.mode == LEARNED_TOKEN_TVI_MODE:
            return self.base.unsqueeze(0).expand(tvi.shape[0], -1)
        tvi = tvi.to(device=self.base.device)
        torch._assert_async(torch.isfinite(tvi).all(), "TVI tensor must contain only finite values")
        if self.mode == METRIC_CAMERA_POSE_TVI_MODE:
            time_features, position_features, rotation_features = metric_camera_pose_features(
                tvi,
                time_wavelengths=self.metric_time_wavelengths,
                position_wavelengths=self.metric_position_wavelengths,
            )
            time_token = self.metric_time_mlp(time_features.to(dtype=self.base.dtype))
            position_token = self.metric_position_mlp(position_features.to(dtype=self.base.dtype))
            rotation_token = self.metric_rotation_mlp(rotation_features.to(dtype=self.base.dtype))
            return (
                self.base.unsqueeze(0)
                + self.metric_time_gate * time_token
                + self.metric_position_gate * position_token
                + self.metric_rotation_gate * rotation_token
            )

        tvi = tvi.to(dtype=self.base.dtype)
        time_features = sinusoidal_scalar_pe(tvi[:, 0], self.hidden_size)
        yaw = tvi[:, 1] if self.mode == TIME_YAW_TVI_MODE else tvi[:, 4]
        angle_half_dim = self.hidden_size // 2
        angle_features = torch.cat(
            [
                sinusoidal_scalar_pe(torch.cos(yaw), angle_half_dim),
                sinusoidal_scalar_pe(torch.sin(yaw), angle_half_dim),
            ],
            dim=-1,
        )
        embeddings = self.base.unsqueeze(0) + self.time_mlp(time_features) + self.angle_mlp(angle_features)
        if self.mode == TIME_CAMERA_POSE_TVI_MODE:
            pose_features = torch.cat(
                [
                    sinusoidal_scalar_pe(tvi[:, 1], 64),
                    sinusoidal_scalar_pe(tvi[:, 2], 64),
                    sinusoidal_scalar_pe(tvi[:, 3], 64),
                    torch.sin(tvi[:, 5]).unsqueeze(-1),
                    torch.cos(tvi[:, 5]).unsqueeze(-1),
                    torch.sin(tvi[:, 6]).unsqueeze(-1),
                    torch.cos(tvi[:, 6]).unsqueeze(-1),
                ],
                dim=-1,
            )
            embeddings = embeddings + self.pose_mlp(pose_features)
        return embeddings

    def replace_masked_rows(self, embeddings: torch.Tensor, row_mask) -> torch.Tensor:
        if not hasattr(self, "mask_token"):
            raise RuntimeError("TVI mask token is disabled for this embedding module")
        if embeddings.ndim != 2:
            raise ValueError(f"TVI embeddings must have shape [N, H], got {tuple(embeddings.shape)}")
        if embeddings.shape[-1] != self.hidden_size:
            raise ValueError(
                f"TVI embedding hidden dimension must be {self.hidden_size}, got {embeddings.shape[-1]}"
            )
        row_mask = torch.as_tensor(row_mask, device=embeddings.device)
        if row_mask.ndim != 1 or row_mask.shape[0] != embeddings.shape[0]:
            raise ValueError(
                f"TVI row mask must have shape [{embeddings.shape[0]}], got {tuple(row_mask.shape)}"
            )
        row_mask = row_mask.to(dtype=torch.bool)
        mask_token = self.mask_token.to(device=embeddings.device, dtype=embeddings.dtype)
        return torch.where(row_mask.unsqueeze(-1), mask_token.unsqueeze(0), embeddings)


__all__ = [
    "TIME_YAW_TVI_MODE",
    "TIME_CAMERA_POSE_TVI_MODE",
    "METRIC_CAMERA_POSE_TVI_MODE",
    "LEARNED_TOKEN_TVI_MODE",
    "CAMERA_POSE_TVI_MODES",
    "get_tvi_input_dim",
    "uses_camera_pose_tvi",
    "NavVLATVIEmbedding",
    "sinusoidal_scalar_pe",
    "metric_camera_pose_features",
]
