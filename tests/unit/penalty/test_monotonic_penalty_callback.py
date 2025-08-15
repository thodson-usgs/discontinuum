import torch
import gpytorch
from rating_gp.models.gpytorch import RatingGPMarginalGPyTorch


def test_monotonic_penalty_callback_smoke():
    n = 12
    torch.manual_seed(0)
    time = torch.linspace(0, 1, n)
    stage = torch.linspace(1.0, 2.5, n)
    X = torch.stack([time, stage], dim=1)
    y = torch.sin(time) + 0.1 * torch.randn(n)

    model = RatingGPMarginalGPyTorch()
    model.model = model.build_model(X, y)

    def penalty_cb():
        device = next(model.model.parameters()).device
        grid_size = 8
        u_time = torch.rand((grid_size,), dtype=torch.float32, device=device)
        time_grid = u_time * (X[:, 0].max() - X[:, 0].min()) + X[:, 0].min()
        eps = 1e-6
        log_xmin = float(torch.log(X[:, 1].min() + eps))
        log_xmax = float(torch.log(X[:, 1].max() + eps))
        u_stage = torch.rand((grid_size,), dtype=torch.float32, device=device)
        log_stage_grid = u_stage * (log_xmax - log_xmin) + log_xmin
        stage_grid = torch.exp(log_stage_grid).requires_grad_(True)
        x_grid = torch.stack([time_grid, stage_grid], dim=1)
        model.model.eval()
        try:
            model.likelihood.eval()
        except Exception:
            pass
        with gpytorch.settings.fast_pred_var():
            with torch.enable_grad():
                mean = model.model(x_grid).mean
        d_mean_d_stage = torch.autograd.grad(mean.sum(), stage_grid, create_graph=True)[0]
        neg = torch.clamp(-d_mean_d_stage, min=0.0)
        pen = neg.mean()
        return pen

    pen = penalty_cb()
    assert torch.is_tensor(pen)
    assert pen.ndim == 0
    assert pen.item() >= 0.0
