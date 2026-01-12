import pytest

__all__ = ["pytest"]

from zhai2022.fokker_planck.gaussian import Gaussian


def test_gaussian():
    import torch

    model = Gaussian(
        potential=lambda r2: (r2 - 1) ** 2,
        rotational_matrix=lambda x: torch.stack(
            [
                torch.stack([torch.zeros(x.shape[0]), torch.ones(x.shape[0])], dim=-1),
                torch.stack([-torch.ones(x.shape[0]), torch.zeros(x.shape[0])], dim=-1),
            ],
            dim=-2,
        ),
        sigma=2.0,
        d_U=lambda r2: 2.0 * (r2 - 1),
        d2_U=lambda r2: 2.0 * torch.ones_like(r2),
    )
    u_t = model.u_t(lambda x, _t: model.steady_state(x), torch.rand(5, 2), 0.0)

    assert u_t.shape == (5,)
    assert torch.norm(u_t) < 1e-5
