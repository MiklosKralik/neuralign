import copy
import math

import numpy as np
import pytest
import torch

import neuralign.ops as ops
from neuralign import (
    NeuralignConfig,
    NeuralignFactors,
    NeuralignModel,
    TensorFactors,
)


def make_source(*, conditions=False, non_negative=False):
    torch.manual_seed(7)
    rank, time = 2, 5
    if non_negative:
        w = torch.tensor([1.2, 0.8])
        V = torch.rand(time, rank) + 0.2
        H = torch.rand(rank, rank) + 0.2
        A = torch.rand(4, rank) + 0.2
        U = torch.rand(4, rank) + 0.2
    else:
        w = torch.tensor([1.2, 0.8])
        V = torch.randn(time, rank)
        H = torch.tensor([[1.0, 0.2], [-0.1, 0.9]])
        A = torch.randn(4, rank)
        U = torch.randn(4, rank)
    Q = torch.linalg.qr(torch.randn(4, rank), mode="reduced").Q
    C = (
        torch.tensor([[1.0, 0.2], [0.3, 1.1], [0.8, 0.7]])
        if conditions
        else None
    )
    return NeuralignFactors(
        w=w,
        V=V,
        H=H,
        tensor_factors=[
            TensorFactors(A=A, U=U, Q=Q, S=torch.ones(rank))
        ],
        C=C,
        condition_labels=["left", "right", "other"] if conditions else None,
        config=NeuralignConfig(
            rank=rank,
            max_iter=6,
            patience=20,
            progress=False,
            seed=11,
            non_negative=non_negative,
        ),
        metadata={"origin": "test"},
    )


def make_tensor(source, neurons=5, trials=6, *, non_negative=False):
    rank = len(source.w)
    if non_negative:
        A = torch.rand(neurons, rank) + 0.2
        U = torch.rand(trials, rank) + 0.2
    else:
        A = torch.randn(neurons, rank)
        U = torch.randn(trials, rank)
    return ops.reconstruct_tensor(source.w, A, source.V, U)


def assert_finite_factors(factors):
    for tensor in (factors.A, factors.U, factors.Q, factors.S):
        assert torch.isfinite(tensor).all()


def test_partial_fit_appends_single_session_without_mutating_source():
    source = make_source()
    original = copy.deepcopy(source)
    tensor = make_tensor(source).requires_grad_()

    result = source.partial_fit(tensor)

    assert result is not source
    assert len(source) == 1
    assert len(result) == 2
    assert tensor.grad is None
    for name in ("w", "V", "H"):
        assert torch.equal(getattr(source, name), getattr(original, name))
        assert torch.equal(getattr(result, name), getattr(original, name))
    assert source.metadata == original.metadata
    assert_finite_factors(result.tensor_factors[-1])
    assert result[-1].reconstructed_tensor().shape == tensor.shape
    adaptation = result.metadata["adaptations"][-1]
    assert adaptation["session_indices"] == [1]
    assert len(adaptation["loss"]) == source.config.max_iter
    assert all(math.isfinite(value) for value in adaptation["loss"])
    assert adaptation["config"]["gauge_fix"] is False


def test_partial_fit_accepts_batched_numpy_and_torch_inputs():
    source = make_source()
    first = make_tensor(source, neurons=4, trials=3).numpy()
    second = make_tensor(source, neurons=6, trials=5)

    result = source.partial_fit([first, second], max_iter=2)

    assert len(result) == 3
    assert result.tensor_factors[-2].A.shape == (4, 2)
    assert result.tensor_factors[-2].U.shape == (3, 2)
    assert result.tensor_factors[-1].A.shape == (6, 2)
    assert result.tensor_factors[-1].U.shape == (5, 2)
    assert result.metadata["adaptations"][-1]["session_indices"] == [1, 2]


def test_condition_alignment_keeps_prototypes_frozen_and_falls_back_per_session():
    source = make_source(conditions=True)
    original_c = source.C.clone()
    tensors = [
        make_tensor(source, neurons=4, trials=4),
        make_tensor(source, neurons=5, trials=3),
        make_tensor(source, neurons=6, trials=3),
    ]
    labels = [
        ["left", "unknown", "right", "unknown"],
        None,
        ["unknown", "unknown", "unknown"],
    ]

    with pytest.warns(UserWarning, match="unknown"):
        result = source.partial_fit(tensors, trial_labels=labels, max_iter=3)

    assert torch.equal(source.C, original_c)
    assert torch.equal(result.C, original_c)
    assert result.metadata["adaptations"][-1]["ignored_labels"] == ["unknown"]
    for factors in result.tensor_factors[-3:]:
        assert_finite_factors(factors)
        assert torch.allclose(
            factors.Q.T @ factors.Q,
            torch.eye(len(source.w)),
            atol=1e-5,
            rtol=1e-5,
        )


def test_condition_centroid_regularizer_updates_only_adapted_trial_factors():
    source = make_source(conditions=True)
    original_c = source.C.clone()
    tensor = make_tensor(source, trials=6)
    labels = ["left", "right", "left", "right", "left", "right"]

    result = source.partial_fit(
        tensor,
        trial_labels=labels,
        trial_mask=[False, False, False, False, True, True],
        lambda_tensor_reconstruction=0.0,
        lambda_procrustes=0.0,
        lambda_condition_centroid=1.0,
        lr=5e-2,
        max_iter=20,
        patience=30,
    )

    adaptation = result.metadata["adaptations"][-1]
    condition_history = adaptation["losses"]["Conditions"]
    assert condition_history[-1] < condition_history[0]
    assert adaptation["config"]["lambda_condition_centroid"] == 1.0
    assert torch.equal(source.C, original_c)
    assert torch.equal(result.C, original_c)
    assert_finite_factors(result.tensor_factors[-1])


def test_supervised_adaptation_enables_small_condition_centroid_default():
    source = make_source(conditions=True)
    tensor = make_tensor(source, trials=4)
    labels = ["left", "right", "left", "right"]

    automatic = source.partial_fit(tensor, trial_labels=labels, max_iter=1)
    disabled = source.partial_fit(
        tensor,
        trial_labels=labels,
        lambda_condition_centroid=0.0,
        max_iter=1,
    )
    unsupervised = source.partial_fit(tensor, max_iter=1)

    automatic_metadata = automatic.metadata["adaptations"][-1]
    disabled_metadata = disabled.metadata["adaptations"][-1]
    unsupervised_metadata = unsupervised.metadata["adaptations"][-1]
    assert automatic_metadata["config"]["lambda_condition_centroid"] == 1e-2
    assert "Conditions" in automatic_metadata["losses"]
    assert disabled_metadata["config"]["lambda_condition_centroid"] == 0.0
    assert "Conditions" not in disabled_metadata["losses"]
    assert unsupervised_metadata["config"]["lambda_condition_centroid"] == 0.0
    assert "Conditions" not in unsupervised_metadata["losses"]


def test_condition_centroid_regularizer_requires_recognized_unmasked_labels():
    source = make_source(conditions=True)
    tensor = make_tensor(source)

    with pytest.raises(ValueError, match="recognized, unmasked trial label"):
        source.partial_fit(
            tensor,
            lambda_condition_centroid=1.0,
            max_iter=1,
        )


def test_random_initialization_is_seeded_and_config_dictionary_is_supported():
    source = make_source()
    tensor = make_tensor(source)
    config = {
        "rank": 2,
        "init": "random",
        "seed": 123,
        "max_iter": 1,
        "progress": False,
    }

    first = source.partial_fit(tensor, config=config)
    second = source.partial_fit(tensor, config=config)

    for name in ("A", "U", "Q", "S"):
        assert torch.equal(
            getattr(first.tensor_factors[-1], name),
            getattr(second.tensor_factors[-1], name),
        )


@pytest.mark.parametrize("loss_type", ["frobenius", "poisson", "poisson_log"])
def test_non_negative_adaptation_supports_all_reconstruction_losses(loss_type):
    source = make_source(non_negative=True)
    tensor = make_tensor(source, non_negative=True)
    if loss_type in {"poisson", "poisson_log"}:
        tensor = torch.poisson(tensor.clamp_min(0.1))

    result = source.partial_fit(
        tensor,
        loss_type=loss_type,
        non_negative=True,
        max_iter=2,
    )

    factors = result.tensor_factors[-1]
    assert_finite_factors(factors)
    for value in (factors.A, factors.U, factors.S):
        assert torch.all(value > 0)


def test_reconstruction_only_fit_improves_the_loss_and_finalizes_q():
    source = make_source()
    tensor = make_tensor(source)

    result = source.partial_fit(
        tensor,
        max_iter=20,
        patience=30,
        lambda_procrustes=0.0,
    )

    history = result.metadata["adaptations"][-1]["loss"]
    assert min(history[1:]) < history[0]
    assert torch.allclose(
        result.tensor_factors[-1].Q.T @ result.tensor_factors[-1].Q,
        torch.eye(2),
        atol=1e-5,
        rtol=1e-5,
    )


def test_alternative_regularizers_and_early_stopping():
    source = make_source()
    tensor = make_tensor(source)
    result = source.partial_fit(
        tensor,
        lambda_tensor_reconstruction=0.0,
        lambda_procrustes=0.0,
        lambda_global_gram=1.0,
        lambda_neuron_sparsity=0.1,
        max_iter=20,
        patience=1,
        tol=1e6,
    )

    adaptation = result.metadata["adaptations"][-1]
    assert set(adaptation["losses"]) == {"Geometry", "Sparsity"}
    assert len(adaptation["loss"]) == 2
    assert_finite_factors(result.tensor_factors[-1])


@pytest.mark.parametrize(
    ("tensor", "message"),
    [
        ([], "one or more"),
        (torch.randn(4, 5), "must have shape"),
        (torch.randn(4, 6, 3), "learned time axis"),
        (torch.randn(4, 5, 0), "unmasked trial"),
    ],
)
def test_invalid_tensor_shapes_are_rejected(tensor, message):
    source = make_source()
    with pytest.raises(ValueError, match=message):
        source.partial_fit(tensor, max_iter=1)


def test_invalid_label_inputs_are_rejected():
    source = make_source()
    tensor = make_tensor(source)
    with pytest.raises(ValueError, match="must match"):
        source.partial_fit(tensor, trial_labels=["too", "short"])
    with pytest.raises(ValueError, match="must match"):
        source.partial_fit(
            tensor, trial_labels=np.zeros((tensor.shape[2], 1))
        )


def test_adaptation_uses_the_frozen_rank():
    source = make_source()
    tensor = make_tensor(source)
    result = source.partial_fit(tensor, rank=3, max_iter=1)

    assert result.metadata["adaptations"][-1]["config"]["rank"] == 2
    assert result.tensor_factors[-1].A.shape[1] == 2


def test_save_load_then_adapt(tmp_path):
    source = make_source(conditions=True)
    path = tmp_path / "factors.pt"
    source.save(path)
    loaded = NeuralignFactors.load(path)

    result = loaded.partial_fit(
        make_tensor(source, trials=3),
        trial_labels=["left", "right", "left"],
        max_iter=2,
    )

    assert len(result) == len(source) + 1
    assert torch.equal(result.C, source.C)
    assert result.condition_labels == source.condition_labels
    assert_finite_factors(result.tensor_factors[-1])


def test_adaptation_integrates_with_joint_model_output():
    tensors = [torch.randn(4, 5, 4), torch.randn(5, 5, 3)]
    model = NeuralignModel(
        tensors,
        config=NeuralignConfig(
            rank=2, max_iter=2, patience=3, progress=False, seed=5
        ),
    )
    source = model.fit("cpu")

    result = source.partial_fit(torch.randn(6, 5, 4), max_iter=2)

    assert len(result) == 3
    assert_finite_factors(result.tensor_factors[-1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_adaptation_can_optimize_on_cuda_and_return_to_source_device():
    source = make_source()
    tensor = make_tensor(source)

    result = source.partial_fit(tensor, device="cuda", max_iter=1)

    for value in (
        result.tensor_factors[-1].A,
        result.tensor_factors[-1].U,
        result.tensor_factors[-1].Q,
        result.tensor_factors[-1].S,
    ):
        assert value.device == source.w.device
