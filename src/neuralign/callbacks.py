"""Minimal callbacks for the training loop."""

import copy
import math

from tqdm.auto import tqdm

from neuralign.gauges import gauge_fix_model


class Callback:
    def on_train_start(self, trainer):
        pass

    def on_step_end(self, trainer):
        pass

    def on_after_optimizer_step(self, trainer):
        pass
    
    def on_train_end(self, trainer):
        pass
    

class EarlyStopping(Callback):
    def __init__(self, patience=30, tol=1e-6):
        self.patience = patience
        self.tol = tol

    def on_train_start(self, trainer):
        self.best = float("inf")
        self.best_loss = float("inf")
        self.best_state = None
        self.best_history_length = 0
        self.bad_steps = 0

    def on_step_end(self, trainer):
        if not math.isfinite(trainer.current_loss):
            trainer.should_stop = True
        else:
            if trainer.current_loss < self.best_loss:
                self.best_loss = trainer.current_loss
                self.best_state = copy.deepcopy(trainer.model.state_dict())
                self.best_history_length = len(trainer.history)

            if trainer.current_loss < self.best - self.tol:
                self.best = trainer.current_loss
                self.bad_steps = 0
            else:
                self.bad_steps += 1
                if self.bad_steps >= self.patience:
                    trainer.should_stop = True

    def on_train_end(self, trainer):
        if self.best_state is not None:
            trainer.model.load_state_dict(self.best_state)
            del trainer.history[self.best_history_length:]
            for history in trainer.loss_history.values():
                del history[self.best_history_length:]


class ProgressBar(Callback):
    def on_train_start(self, trainer):
        self.bar = tqdm(total=trainer.max_steps)

    def on_step_end(self, trainer):
        self.bar.update(1)
        postfix = {"loss": f"{trainer.current_loss:.6f}"}
        postfix.update(
            (f"{name}", f"{value:.6f}")
            for name, value in trainer.current_losses.items()
        )
        self.bar.set_postfix(postfix)

    def on_train_end(self, trainer):
        self.bar.close()
        
        
class GaugeFixing(Callback):
    def on_train_start(self, trainer):
        # Fix the gauge of the decomposition into the global weights at the start of training
        gauge_fix_model(trainer.model)
    
    def on_after_optimizer_step(self, trainer):
        # Fix the gauge of the decomposition into the global weights after each step
        gauge_fix_model(trainer.model)
        
    def on_train_end(self, trainer):
        # Also resolve sign ambiguities when the decomposition is finalized.
        gauge_fix_model(trainer.model, sign_fix=True)
