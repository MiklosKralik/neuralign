import warnings

import numpy as np
import pytest
import torch

import neuralign.ops as ops
from neuralign import NeuralignConfig, NeuralignFactors, NeuralignModel, TensorFactors


def config(rank=2):
    return NeuralignConfig(
        rank=rank,
        max_iter=2,
        patience=4,
        progress=False,
        seed=7,
    )


def source_factors(*, conditions=False):
    torch.manual_seed(4)
    rank, time = 2, 5
    w = torch.tensor([1.0, 0.8])
    V = torch.randn(time, rank)
    H = torch.tensor([[1.0, 0.2], [-0.1, 0.9]])
    A = torch.randn(4, rank)
    U = torch.randn(4, rank)
    Q = torch.linalg.qr(torch.randn(4, rank), mode="reduced").Q
    C = torch.tensor([[1.0, 0.2], [0.3, 1.1]]) if conditions else None
    return NeuralignFactors(
        w=w,
        V=V,
        H=H,
        tensor_factors=[TensorFactors(A=A, U=U, Q=Q, S=torch.ones(rank))],
        C=C,
        condition_labels=["left", "right"] if conditions else None,
        config=config(rank),
        metadata={},
    )


def make_tensor(source, neurons=5, trials=5):
    A = torch.randn(neurons, len(source.w))
    U = torch.randn(trials, len(source.w))
    return ops.reconstruct_tensor(source.w, A, source.V, U)


def test_joint_unsupervised_mask_only_changes_alignment_slice():
    tensors = [torch.randn(4, 5, 4), torch.randn(5, 5, 5)]
    masks = [
        np.array([False, True, False, True]),
        torch.tensor([True, False, False, True, False]),
    ]

    model = NeuralignModel(tensors, trial_mask=masks, config=config())

    for tensor, mask, tensor_slice, indices in zip(
        tensors, masks, model.tensor_slices, model.alignment_indices
    ):
        keep = ~torch.as_tensor(mask)
        assert torch.equal(indices, torch.nonzero(keep, as_tuple=True)[0])
        assert torch.allclose(tensor_slice, tensor[:, :, keep].mean(dim=2))

    factors = model.fit("cpu")
    assert [session.U.shape[0] for session in factors.tensor_factors] == [4, 5]
    assert [factors[n].reconstructed_tensor().shape for n in range(len(factors))] == [
        tensor.shape for tensor in tensors
    ]


def test_joint_condition_vocabulary_omits_masked_only_conditions():
    tensor = torch.randn(4, 5, 5)
    labels = [["left", "right", "masked_only", "left", "masked_only"]]
    mask = [[False, False, True, False, True]]

    model = NeuralignModel(
        [tensor],
        trial_labels=labels,
        trial_mask=mask,
        config=config(),
    )

    assert model.condition_labels == ["left", "right"]
    assert model.C.shape == (2, 2)
    used_trials = torch.cat([indices for _, _, indices in model.label_info])
    assert set(used_trials.tolist()) == {0, 1, 3}


def test_omitted_joint_mask_matches_all_false_mask():
    tensors = [torch.randn(4, 5, 4), torch.randn(5, 5, 3)]
    default = NeuralignModel(tensors, config=config())
    explicit = NeuralignModel(
        tensors,
        trial_mask=[
            np.zeros(tensor.shape[2], dtype=bool) for tensor in tensors
        ],
        config=config(),
    )

    for name in ("w", "V", "H"):
        assert torch.allclose(getattr(default, name), getattr(explicit, name))
    for n in range(len(tensors)):
        for name in ("A", "U", "Q_n", "S_n"):
            assert torch.allclose(
                getattr(default, name)[n], getattr(explicit, name)[n]
            )


def test_adaptation_mask_excludes_trials_from_unsupervised_alignment():
    source = source_factors()
    tensor = make_tensor(source)
    changed = tensor.clone()
    changed[:, :, -1] += 1000
    mask = torch.tensor([False, False, False, False, True])
    kwargs = dict(
        trial_mask=mask,
        lambda_tensor_reconstruction=0.0,
        lambda_procrustes=1.0,
        max_iter=2,
    )

    first = source.partial_fit(tensor, **kwargs).tensor_factors[-1]
    second = source.partial_fit(changed, **kwargs).tensor_factors[-1]

    for name in ("A", "Q", "S"):
        assert torch.allclose(getattr(first, name), getattr(second, name))
    assert torch.allclose(first.U[:-1], second.U[:-1])
    assert first.U.shape[0] == tensor.shape[2]


def test_masked_adaptation_labels_are_not_used_or_reported():
    source = source_factors(conditions=True)
    tensor = make_tensor(source, trials=3)
    original_c = source.C.clone()

    with warnings.catch_warnings(record=True) as caught:
        result = source.partial_fit(
            tensor,
            trial_labels=["left", "right", "unknown"],
            trial_mask=[False, False, True],
            max_iter=2,
        )

    assert caught == []
    assert result.metadata["adaptations"][-1]["ignored_labels"] == []
    assert torch.equal(result.C, original_c)
    assert result.tensor_factors[-1].U.shape[0] == 3


def test_omitted_adaptation_mask_matches_all_false_mask():
    source = source_factors()
    tensor = make_tensor(source)

    default = source.partial_fit(tensor, max_iter=2).tensor_factors[-1]
    explicit = source.partial_fit(
        tensor,
        trial_mask=np.zeros(tensor.shape[2], dtype=bool),
        max_iter=2,
    ).tensor_factors[-1]

    # Repeated SVD/Procrustes calls can vary slightly across BLAS backends.
    for name in ("A", "U", "Q", "S"):
        default_value = getattr(default, name)
        explicit_value = getattr(explicit, name)
        relative_error = torch.linalg.vector_norm(
            default_value - explicit_value
        ) / torch.linalg.vector_norm(default_value).clamp_min(1.0)
        assert relative_error < 1e-2


@pytest.mark.parametrize("where", ["joint", "adapt"])
def test_all_masked_sessions_are_rejected(where):
    tensor = torch.randn(4, 5, 3)
    mask = [True, True, True]

    with pytest.raises(ValueError, match="unmasked trial"):
        if where == "joint":
            NeuralignModel([tensor], trial_mask=[mask], config=config())
        else:
            source_factors().partial_fit(tensor, trial_mask=mask, max_iter=1)


@pytest.mark.parametrize("where", ["joint", "adapt"])
def test_trial_mask_lengths_must_match_trials(where):
    tensor = torch.randn(4, 5, 3)
    mask = [False, True]

    with pytest.raises(ValueError, match="must match"):
        if where == "joint":
            NeuralignModel([tensor], trial_mask=[mask], config=config())
        else:
            source_factors().partial_fit(tensor, trial_mask=mask, max_iter=1)
