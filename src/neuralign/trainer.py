"""Minimal optimization loop for Neuralign models."""

import torch


class Trainer:
    def __init__(
        self,
        model,
        losses,
        lr=1e-2,
        max_steps=200,
        callbacks=None,
    ):
        self.model = model
        self.losses = losses
        self.lr = lr
        self.max_steps = max_steps
        self.callbacks = list(callbacks or [])
        self.history = []
        self.loss_history = {name: [] for name in self.losses}
        self.current_loss = float("inf")
        self.current_losses = {
            name: float("inf") for name in self.losses
        }
        self.should_stop = False

    def fit(self):
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(parameters, lr=self.lr)

        for callback in self.callbacks:
            callback.on_train_start(self)

        try:
            for step in range(self.max_steps):
                self.optimizer.zero_grad(set_to_none=True)
                component_losses = {
                    name: loss(self.model)
                    for name, (loss, _) in self.losses.items()
                }
                for name, value in component_losses.items():
                    if value.numel() != 1:
                        raise ValueError(
                            f"Loss {name!r} must return a scalar, got shape "
                            f"{tuple(value.shape)}"
                        )
                total_loss = torch.stack(tuple(
                    weight * component_losses[name]
                    for name, (_, weight) in self.losses.items()
                )).sum()
                total_loss.backward()
                self.optimizer.step()
                
                for callback in self.callbacks:
                    callback.on_after_optimizer_step(self)

                self.step = step
                self.current_losses = {
                    name: value.detach().item()
                    for name, value in component_losses.items()
                }
                self.current_loss = total_loss.detach().item()
                self.history.append(self.current_loss)
                for name, value in self.current_losses.items():
                    self.loss_history[name].append(value)

                for callback in self.callbacks:
                    callback.on_step_end(self)

                if self.should_stop:
                    break
        except KeyboardInterrupt: # User can stop training with Ctrl+C
            print("Training interrupted by user.")
        finally:
            for callback in self.callbacks:
                callback.on_train_end(self)

        self.model.finalize(self)
        
        return self.model.export_factors(self.history, self.loss_history)
