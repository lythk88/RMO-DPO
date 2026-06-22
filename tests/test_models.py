from __future__ import annotations

from rmo_dpo.models import can_reuse_peft_base_as_reference


class DummyPeftModel:
    def disable_adapter(self):
        raise NotImplementedError


class DummyPlainModel:
    pass


def test_reuse_peft_reference_only_when_reference_matches_policy():
    model = DummyPeftModel()
    assert can_reuse_peft_base_as_reference(
        use_lora=True,
        policy_name="policy",
        reference_name="policy",
        model=model,
    )
    assert not can_reuse_peft_base_as_reference(
        use_lora=True,
        policy_name="policy",
        reference_name="different-ref",
        model=model,
    )


def test_reuse_peft_reference_requires_disable_adapter():
    assert not can_reuse_peft_base_as_reference(
        use_lora=True,
        policy_name="policy",
        reference_name="policy",
        model=DummyPlainModel(),
    )
    assert not can_reuse_peft_base_as_reference(
        use_lora=False,
        policy_name="policy",
        reference_name="policy",
        model=DummyPeftModel(),
    )
