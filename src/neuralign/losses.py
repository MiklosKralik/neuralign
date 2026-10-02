"""Losses used to train Neuralign models."""

import torch

import neuralign.ops as ops

def relative_fro_error_sq(pred, target, target_norm_sq):
    """||pred-target||_F^2 / ||target||_F^2.

    target_norm_sq can be precomputed for speed.
    """
    return torch.linalg.norm(pred - target) ** 2 / target_norm_sq

def poisson_nll_loss(pred, target, log_input=False, eps=1e-8):
    return torch.nn.functional.poisson_nll_loss(pred, target, log_input=log_input, eps=eps)

def symmetric_fro_error_sq(pred, target, eps=1e-12):
    """||pred-target||_F^2 / 1/2(||pred||_F^2 + ||target||_F^2).

    This is a symmetric version of relative_fro_error_sq that is more numerically stable when the target is close to zero.
    """
    pred_minus_target_norm_sq = torch.linalg.norm(pred - target) ** 2
    pred_norm_sq = torch.linalg.norm(pred) ** 2
    target_norm_sq = torch.linalg.norm(target) ** 2
    return pred_minus_target_norm_sq / (0.5 * (pred_norm_sq + target_norm_sq) + eps).detach()

class TensorReconstructionLoss(torch.nn.Module):
    """Mean relative squared reconstruction error across tensors."""

    def __init__(self, norms_squared, loss_type):
        super().__init__()
        self.norms_squared = norms_squared # precomputed Frobenius norms of the target tensors for efficiency (on correct device)
        if loss_type == "frobenius":
            self.loss_fn = self.frobenius_loss
        elif loss_type == "poisson":
            self.loss_fn = self.poisson_loss
        elif loss_type == "poisson_log":
            self.loss_fn = self.poisson_log_loss
        else:
            raise ValueError(f"Invalid loss_type: {loss_type}")
        
    def frobenius_loss(self, pred, target, n):
        return relative_fro_error_sq(pred, target, self.norms_squared[n])
    
    def poisson_loss(self, pred, target, n):
        return poisson_nll_loss(pred, target)
    
    def poisson_log_loss(self, pred, target, n):
        return poisson_nll_loss(pred, target, log_input=True)

    def forward(self, model):
            
        loss = torch.tensor(0.0, device=model.tensors[0].device, dtype=model.tensors[0].dtype)
        for n, tensor in enumerate(model.tensors):
            pred = ops.reconstruct_tensor(
                model.w, model.A[n], model.V, model.U[n]
            )
            loss += self.loss_fn(pred, tensor, n)
            
        return loss / len(model.tensors)

class GlobalGramLoss(torch.nn.Module):
    """ Relative squared error of the global geometry vs. the decpmposition estimated geometry
    || A_n^T A_n - diag(S_n)H^THdiag(S_n) ||_F^2 / || diag(S_n)H^THdiag(S_n) ||_F^2
    """
    
    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps # small constant to avoid division by zero in relative error calculation
        
    def forward(self, model):
        loss = torch.tensor(0.0, device=model.tensors[0].device, dtype=model.tensors[0].dtype)
        for n, _ in enumerate(model.tensors):
            A_gram = ops.gram_from_A(model.A[n])
            H_gram = ops.gram_from_H(model.H, model.S_n[n])
            loss += symmetric_fro_error_sq(A_gram, H_gram, eps=self.eps)
        return loss / len(model.tensors)
    
class GlobalProcrustesLoss(torch.nn.Module):
    """
    1. Update projections by solving orthogonal procrustes problem Q_n = argmin_Q_n || Q_n^TX_n - [w, S_n, H, V, N_n] ||
    
    2. Use soft loss between Q_n and A_n --> ||A_n - Q_ndiag(S_n)H ||
    
    """
    
    def __init__(self, model, k=1, eps=1e-12):
        super().__init__()
        self.k = k # how often to re-compute Q_n (every k iterations)
        self.iter = 0
        self.eps = eps # small constant to avoid division by zero in relative error calculation
        
        # tensor_slices contain averages over alignment trials only
        for n, trial_avg in enumerate(model.tensor_slices):
            self.register_buffer(
                f"TrialAvg_{n}", trial_avg.detach().clone(), persistent=False
            )
    
    @torch.no_grad()
    def update_projections(self, model):
        for n, _ in enumerate(model.tensors):
            trial_avg = getattr(self, f"TrialAvg_{n}") # shape (neurons, time)
            trial_factors = model.U[n].index_select(
                0, model.alignment_indices[n]
            )
            pred_avg_latents = ops.predicted_latent_trajectories(
                model.w,
                model.H,
                model.S_n[n],
                model.V,
                trial_factors.mean(dim=0, keepdim=True),
            ).squeeze(0) # shape (time, rank)
            model.Q_n[n].copy_(ops.solve_procrustes(trial_avg, pred_avg_latents.T))
    
    def forward(self, model):
        if self.iter % self.k == 0:
            self.update_projections(model)
        
        loss = torch.tensor(0.0, device=model.tensors[0].device, dtype=model.tensors[0].dtype)
        
        for n, _ in enumerate(model.tensors):
            A_aligned = model.Q_n[n] @ ops.scale_mixing(model.H, model.S_n[n])
            loss += relative_fro_error_sq(model.A[n], A_aligned, torch.linalg.norm(A_aligned) ** 2 + self.eps)

        self.iter += 1

        return loss / len(model.tensors)
    
class ConditionProcrustesLoss(torch.nn.Module):
    """
    1. Stack condition-means of the neural activity and the predicted condition-mean latents
    
    2. Update projections based on the stacked representaions just like in the global procrustes loss
    
    3. Use soft loss between Q_n and A_n --> ||A_n - Q_ndiag(S_n)H ||
    """
    
    def __init__(self, model, k=1, eps=1e-12):
        super().__init__()
        self.k = k # how often to re-compute Q_n (every k iterations)
        self.iter = 0
        self.eps = eps # small constant to avoid division by zero in relative error calculation
        
        # register condition-averaged stacked slices as buffers for each tensor
        for n, _ in enumerate(model.tensors):
            stacked_condition_avg = self._stack_condition_averages(model, n)
            if stacked_condition_avg is not None:
                self.register_buffer(f"CondAvg_{n}", stacked_condition_avg, persistent=False)
    
    def _stack_condition_averages(self, model, n):
        # get the stacked condition-averaged neural activity for each tensor n
        blocks = []
        
        for _, cond, _  in model.session_label_info[n]:
            cond_indices = getattr(model, f"_label_info_{n}_{cond}")
            trial_avg_nc = torch.mean(model.tensors[n].index_select(2, cond_indices), dim=2) # shape (neurons, time)
            blocks.append(trial_avg_nc)
            
        if not blocks:
            return None
        
        return torch.cat(blocks, dim=1) # shape (neurons, time * n_conditions)
    
    def _stack_condition_factors(self, model, n):
        # get the stacked condition-averaged predicted latents for each tensor n
        blocks = []
        
        for _, cond, _ in model.session_label_info[n]:
            pred_avg_latents_nc = ops.predicted_latent_trajectories(
                model.w, model.H, model.S_n[n], model.V, model.C[cond].unsqueeze(0),
            ).squeeze(0) # shape (time, rank)
            blocks.append(pred_avg_latents_nc)
            
        if not blocks:
            return None
        
        return torch.cat(blocks, dim=0) # shape (time * n_conditions, rank)
    
    @torch.no_grad()
    def _update_condition_centroids(self, model):
        # update the condition centroids based on the current trial factors
        C = torch.zeros_like(model.C)
        counts = torch.zeros(model.C.shape[0], device=model.C.device, dtype=model.C.dtype)
        for n, cond, _ in model.label_info:
            cond_indices = getattr(model, f"_label_info_{n}_{cond}")
            C[cond] += torch.mean(model.U[n].index_select(0, cond_indices), dim=0)
            counts[cond] += 1
        model.C.copy_(C / counts.unsqueeze(1))
    
    @torch.no_grad()
    def update_projections(self, model):
        for n, _ in enumerate(model.tensors):
            cond_avg = getattr(self, f"CondAvg_{n}") # shape (neurons, time)
            pred_avg_latents = self._stack_condition_factors(model, n) # shape (time, rank)
            model.Q_n[n].copy_(ops.solve_procrustes(cond_avg, pred_avg_latents.T))
                
    def forward(self, model):
        if self.iter % self.k == 0:
            self._update_condition_centroids(model)
            self.update_projections(model)
        
        loss = torch.tensor(0.0, device=model.tensors[0].device, dtype=model.tensors[0].dtype)
        
        for n, _ in enumerate(model.tensors):
            A_aligned = model.Q_n[n] @ ops.scale_mixing(model.H, model.S_n[n])
            loss += relative_fro_error_sq(model.A[n], A_aligned, torch.linalg.norm(A_aligned) ** 2 + self.eps)

        self.iter += 1

        return loss / len(model.tensors)


class AdaptationProcrustesLoss(torch.nn.Module):
    """Align new sessions without changing learned condition prototypes."""

    def __init__(self, model, session_label_info, k=1, eps=1e-12):
        super().__init__()
        self.k = k
        self.iter = 0
        self.eps = eps
        self.session_label_info = session_label_info
        self.condition_indices = []

        for n, condition_info in enumerate(session_label_info):
            condition_ids = tuple(cond for cond, _ in condition_info)
            self.condition_indices.append(condition_ids)
            if condition_info:
                blocks = [
                    model.tensors[n].index_select(2, indices).mean(dim=2)
                    for _, indices in condition_info
                ]
                observed = torch.cat(blocks, dim=1)
            else:
                observed = model.tensors[n].index_select(
                    2, model.alignment_indices[n]
                ).mean(dim=2)
            self.register_buffer(
                f"Observed_{n}", observed.detach().clone(), persistent=False
            )

    def _predicted_coordinates(self, model, n):
        condition_ids = self.condition_indices[n]
        if condition_ids:
            blocks = [
                ops.predicted_latent_trajectories(
                    model.w,
                    model.H,
                    model.S_n[n],
                    model.V,
                    model.C[cond].unsqueeze(0),
                ).squeeze(0)
                for cond in condition_ids
            ]
            return torch.cat(blocks, dim=0)

        trial_factors = model.U[n].index_select(
            0, model.alignment_indices[n]
        )
        return ops.predicted_latent_trajectories(
            model.w,
            model.H,
            model.S_n[n],
            model.V,
            trial_factors.mean(dim=0, keepdim=True),
        ).squeeze(0)

    @torch.no_grad()
    def update_projections(self, model):
        for n in range(len(model.tensors)):
            observed = getattr(self, f"Observed_{n}")
            predicted = self._predicted_coordinates(model, n)
            model.Q_n[n].copy_(ops.solve_procrustes(observed, predicted.T))

    def forward(self, model):
        if self.iter % self.k == 0:
            self.update_projections(model)

        losses = []
        for n in range(len(model.tensors)):
            aligned = model.Q_n[n] @ ops.scale_mixing(
                model.H, model.S_n[n]
            )
            losses.append(
                relative_fro_error_sq(
                    model.A[n],
                    aligned,
                    torch.linalg.norm(aligned) ** 2 + self.eps,
                )
            )

        self.iter += 1
        return torch.stack(losses).mean()


class AdaptationConditionCentroidLoss(torch.nn.Module):
    """Match adapted condition means to frozen source prototypes.

    Only recognized, unmasked labels are included. The source prototypes are
    buffers on the adaptation model, so this loss updates the new trial
    factors without making the shared condition centroids plastic.
    """

    def __init__(self, session_label_info):
        super().__init__()
        self.entries = []
        for session, condition_info in enumerate(session_label_info):
            for condition, indices in condition_info:
                name = f"Indices_{session}_{condition}"
                self.register_buffer(
                    name, indices.detach().clone(), persistent=False
                )
                self.entries.append((session, condition, name))

    def forward(self, model):
        losses = []
        for session, condition, name in self.entries:
            indices = getattr(self, name)
            condition_mean = model.U[session].index_select(0, indices).mean(dim=0)
            losses.append(
                torch.linalg.norm(condition_mean - model.C[condition]) ** 2
            )
        return torch.stack(losses).mean()


class ConditionCentroidLoss(torch.nn.Module):
    """ Mean squared error of the learned shared condition centroids vs. the decomposition estimated centroids
    m_nc : mean of the n-th tensor's trial factors for condition c 
    1/nc sum{||m_nc - C_c||^2}
    assumest condition centroids are implemented as self.C parameterlist
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, model):
        loss = torch.tensor(0.0, device=model.tensors[0].device, dtype=model.tensors[0].dtype)
        for n, cond, cond_indices in model.label_info:
            m_nc = torch.mean(model.U[n].index_select(0, cond_indices), dim=0) # mean of the n-th tensor's trial factors for condition c
            loss += torch.linalg.norm(m_nc - model.C[cond]) ** 2
        return loss / len(model.label_info) # normalize by the number of conditions across all tensors
    
class ConditionGramLoss(torch.nn.Module):
    """
    condition-aware geometry loss:
    
    """
    
    def __init__(self):
        super().__init__()

    def forward(self, model):
        loss = torch.tensor(0.0, device=model.tensors[0].device, dtype=model.tensors[0].dtype)
        
        # pre-calculate grams from neuron factors
        A_grams = [ops.gram_from_A(model.A[n]) for n in range(len(model.A))]
        H_grams = [ops.gram_from_H(model.H, model.S_n[n]) for n in range(len(model.A))]
        
        for n, cond, cond_indices in model.label_info:
            U_nc = torch.diag(torch.mean(model.U[n].index_select(0, cond_indices), dim=0)) # mean of the n-th tensor's trial factors for condition c
            A_gram_nc = U_nc @ A_grams[n] @ U_nc # condition-component geometry for tensor n, condition c
            C_c = torch.diag(model.C[cond]) # learned shared condition mean for condition c
            H_gram_nc = C_c @ H_grams[n] @ C_c # predicted shared component geometry for tensor n, condition c
            
            loss += symmetric_fro_error_sq(A_gram_nc, H_gram_nc)
        return loss / len(model.label_info) # normalize by the number of conditions across all tensors
    
class NeuronSparsityLoss(torch.nn.Module):
    """Scale-invariant column-wise Hoyer sparsity of neuron factors."""

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, model):
        losses = []

        for A in model.A:
            n_neurons = A.shape[0]

            if n_neurons <= 1:
                continue

            l1 = torch.linalg.vector_norm(A, ord=1, dim=0)
            l2 = torch.linalg.vector_norm(A, ord=2, dim=0)

            # At an exactly zero column, assign the maximum dense penalty.
            ratio = torch.where(
                l2 > self.eps,
                l1 / l2.clamp_min(self.eps),
                torch.full_like(l2, n_neurons**0.5),
            )

            normalized = (
                (ratio - 1.0) /
                (n_neurons**0.5 - 1.0)
            )
            losses.append(normalized.mean())

        if not losses:
            return model.A[0].new_zeros(())

        return torch.stack(losses).mean()
