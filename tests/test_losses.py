import torch

from rmo_dpo.losses import (
    clip_and_normalize_weights,
    dpo_loss_from_logps,
    eta_sgd_update,
    raw_dro_weights,
)


def test_dpo_loss_uses_binary_preference_label():
    policy_a = torch.tensor([2.0, 2.0])
    policy_b = torch.tensor([0.0, 0.0])
    ref_a = torch.tensor([0.0, 0.0])
    ref_b = torch.tensor([0.0, 0.0])
    labels = torch.tensor([1.0, 0.0])
    result = dpo_loss_from_logps(policy_a, policy_b, ref_a, ref_b, labels=labels, beta=1.0)
    assert result.losses[0] < result.losses[1]
    assert torch.equal(result.accuracy, torch.tensor([1.0, 0.0]))


def test_kl_weights_are_positive_and_normalized():
    losses = torch.tensor([0.1, 1.0, 2.0])
    raw = raw_dro_weights(losses, eta=0.5, lam=0.7, divergence="kl")
    weights = clip_and_normalize_weights(raw, omega_max=None, normalize=True)
    assert torch.all(weights > 0)
    assert torch.allclose(weights.mean(), torch.tensor(1.0), atol=1e-6)


def test_chi2_weights_nonnegative():
    losses = torch.tensor([-1.0, 0.0, 1.0])
    raw = raw_dro_weights(losses, eta=0.0, lam=0.5, divergence="chi2")
    assert torch.all(raw >= 0)


def test_eta_update_moves_up_when_mean_omega_gt_one():
    eta = eta_sgd_update(eta=0.0, raw_weights=torch.tensor([2.0, 2.0]), eta_lr=0.1, divergence="kl")
    assert eta > 0.0
