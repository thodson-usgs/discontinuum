"""Orthogonal random Fourier feature noise model."""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch


class OrthogonalRFFNoise:
    """Approximate noise using orthogonal random Fourier features.

    This class generates an orthogonal random Fourier feature (RFF) mapping
    which can be used to approximate stationary covariance functions. Each
    input dimension can be scaled individually via the ``scales`` argument.

    Parameters
    ----------
    input_dim : int
        Dimension of the input space.
    num_features : int
        Number of random features to draw.
    scales : Sequence[float], optional
        Per-dimension scaling factors. If ``None``, all dimensions are scaled
        by 1.0.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        input_dim: int,
        num_features: int,
        scales: Optional[Sequence[float]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.input_dim = int(input_dim)
        self.num_features = int(num_features)

        if scales is None:
            scales = torch.ones(self.input_dim, dtype=torch.float32)
        else:
            scales = torch.as_tensor(scales, dtype=torch.float32)
            if scales.numel() != self.input_dim:
                raise ValueError("Length of scales must equal input_dim")
        self.scales = scales

        self.seed = seed
        self._init_features()

    def _init_features(self) -> None:
        gen = torch.Generator()
        if self.seed is not None:
            gen.manual_seed(int(self.seed))

        # Draw random weights in blocks so that within each block the
        # feature vectors are orthogonal. This follows the approach of
        # "Orthogonal Random Features" where the random frequencies are
        # generated from orthogonal matrices.
        d = self.input_dim
        n_blocks = math.ceil(self.num_features / d)
        blocks = []
        for _ in range(n_blocks):
            g = torch.randn(d, d, generator=gen)
            q, _ = torch.linalg.qr(g)
            blocks.append(q)
        w = torch.cat(blocks, dim=1)[:, : self.num_features]

        # Apply per-dimension scaling. Each row corresponds to an input dim.
        self.weight = w / self.scales[:, None]

        # Random phase for each feature
        self.phase = 2 * math.pi * torch.rand(self.num_features, generator=gen)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the orthogonal RFF features for ``x``.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape ``(n_samples, input_dim)``.

        Returns
        -------
        torch.Tensor
            Feature matrix of shape ``(n_samples, num_features)``.
        """
        x = torch.as_tensor(x, dtype=torch.float32)
        proj = x @ self.weight
        return math.sqrt(2.0 / self.num_features) * torch.cos(proj + self.phase)
