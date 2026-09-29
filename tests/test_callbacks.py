from types import SimpleNamespace

import torch

from neuralign.callbacks import EarlyStopping


def test_early_stopping_restores_factors_with_lowest_loss():
    model = torch.nn.Linear(1, 1, bias=False)
    trainer = SimpleNamespace(
        model=model,
        should_stop=False,
        history=[],
        loss_history={"test": []},
    )
    callback = EarlyStopping(patience=2, tol=1.0)
    callback.on_train_start(trainer)

    for weight, loss in [(1.0, 10.0), (2.0, 9.5), (3.0, 9.75)]:
        with torch.no_grad():
            model.weight.fill_(weight)
        trainer.current_loss = loss
        trainer.history.append(loss)
        trainer.loss_history["test"].append(loss / 2)
        callback.on_step_end(trainer)

    assert trainer.should_stop
    callback.on_train_end(trainer)
    assert model.weight.item() == 2.0
    assert trainer.history == [10.0, 9.5]
    assert trainer.loss_history["test"] == [5.0, 4.75]
