"""Tests for repeat-then-shift encoder data schedule."""

from pathologies.experiments.fixed_batch_data import RepeatThenShiftDataset


def test_repeat_then_shift_dataset_steps() -> None:
    """Early steps repeat batch A; the final step uses batch B."""
    repeat = [{"input_ids": [1], "labels": [1]}]
    shift = [{"input_ids": [2], "labels": [2]}]
    dataset = RepeatThenShiftDataset(repeat, shift, num_steps=100)
    assert len(dataset) == 100
    assert dataset[0]["input_ids"] == [1]
    assert dataset[98]["input_ids"] == [1]
    assert dataset[99]["input_ids"] == [2]


def test_repeat_then_tail_on_shift_batch() -> None:
    """After the shift step, the tail keeps serving the shift batch."""
    repeat = [{"input_ids": [1], "labels": [1]}]
    shift = [{"input_ids": [2], "labels": [2]}]
    dataset = RepeatThenShiftDataset(
        repeat,
        shift,
        leading_repeat_steps=99,
        tail_repeat_shift_steps=100,
    )
    assert dataset.num_steps == 200
    assert dataset.shift_step_index == 99
    assert dataset[98]["input_ids"] == [1]
    assert dataset[99]["input_ids"] == [2]
    assert dataset[100]["input_ids"] == [2]
    assert dataset[199]["input_ids"] == [2]


def test_repeat_shift_experiment_registered() -> None:
    """New experiment name is available without adding failure modes."""
    from pathologies.experiment import get_experiment

    experiment = get_experiment("encoder_mlm_repeat_shift")
    assert experiment.profile_family == "encoder"
    assert "adam_collapse" in experiment.failure_modes
    assert "bad_mlm_mask" in experiment.failure_modes
