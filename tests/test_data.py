from rmo_dpo.data import build_attribute_pairs, noise_rates_for_split, parse_helpsteer2_dialogue


def test_parse_single_turn():
    messages = parse_helpsteer2_dialogue("Write a joke", system_message="System")
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "Write a joke"}


def test_parse_multiturn():
    prompt = "Hi<extra_id_1>Assistant\nHello<extra_id_1>User\nHelp me"
    messages = parse_helpsteer2_dialogue(prompt)
    assert messages == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "Help me"},
    ]


def test_build_attribute_pairs_emits_algorithm1_schema():
    rows = [
        {"prompt": "p", "response": "A", "helpfulness": 4},
        {"prompt": "p", "response": "B", "helpfulness": 1},
    ]
    pairs, stats = build_attribute_pairs(rows, split="train", objectives=["helpfulness"])
    row = pairs["helpfulness"][0]
    assert {"prompt", "response_a", "response_b", "preference_label", "objective"}.issubset(row)
    assert row["response_a"] == "A"
    assert row["response_b"] == "B"
    assert row["preference_label"] == 1
    assert stats[0].total_pairs == 1


def test_noise_rates_only_apply_to_train_split():
    noise = {"helpfulness": 0.3}
    assert noise_rates_for_split("train", noise) == noise
    assert noise_rates_for_split("validation", noise) == {}
    assert noise_rates_for_split("test", noise) == {}


def test_validation_pairs_remain_clean_when_noise_requested():
    rows = [
        {"prompt": "p", "response": "A", "helpfulness": 4},
        {"prompt": "p", "response": "B", "helpfulness": 1},
    ]
    pairs, stats = build_attribute_pairs(
        rows,
        split="validation",
        objectives=["helpfulness"],
        noise_rates=noise_rates_for_split("validation", {"helpfulness": 1.0}),
        seed=0,
    )
    row = pairs["helpfulness"][0]
    assert row["preference_label"] == 1
    assert row["label_flipped"] is False
    assert stats[0].flipped_by_noise == 0
