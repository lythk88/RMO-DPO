import torch

from rmo_dpo.conflict import combine_gradients, project_simplex


def test_project_simplex():
    v = torch.tensor([0.2, -1.0, 3.0])
    p = project_simplex(v)
    assert torch.all(p >= 0)
    assert torch.allclose(p.sum(), torch.tensor(1.0), atol=1e-6)


def test_weighted_combination_shape():
    grads = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    result = combine_gradients(
        grads,
        mode="weighted",
        user_weights=torch.tensor([0.25, 0.75]),
    )
    assert result.direction.shape == (2,)
    assert torch.allclose(result.direction, torch.tensor([0.25, 1.5]), atol=1e-6)


def test_mgda_combination_simplex_coefficients():
    grads = torch.randn(3, 5)
    result = combine_gradients(
        grads,
        mode="mgda",
        user_weights=torch.ones(3) / 3,
        qp_steps=10,
    )
    assert torch.all(result.coefficients >= 0)
    assert torch.allclose(result.coefficients.sum(), torch.tensor(1.0), atol=1e-5)
