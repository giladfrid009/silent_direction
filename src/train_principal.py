import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm

from src.utils.torch import extract_device
from src.activation_extractor import ActivationExtractor, ActivationManipulator
from src.model_helpers import tokenize
from src.aliases import Conv
from src.metrics import Metrics
from src.losses import Loss, compute_targets_mask
from src.model_helpers import project


@torch.no_grad()
def compute_empirical_mean(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    layer_name: str,
    conversations: list,
    num_iters: int = 200,
    batch_size: int = 32,
    normalize: bool = False,
) -> torch.Tensor:
    """
    Computes the empirical mean of activations at the specified layer over the given conversations.
    The mean is computed over all token positions and all conversations.

    Returns:
        (torch.Tensor): The empirical mean of activations at the specified layer.
    """
    device = extract_device(model)
    activ_extractor = ActivationExtractor(model, layer_name)

    mean_activ: torch.Tensor = None
    total_count = 0

    num_conversations = len(conversations)
    num_batches = (num_conversations + batch_size - 1) // batch_size
    num_iters = min(num_iters, num_batches)

    print(f"Computing empirical mean over {num_conversations} conversations in {num_iters} batches of size {batch_size}...")

    for batch_idx in tqdm(range(num_iters), leave=False, desc="Empirical Mean"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_conversations)

        batch_convs = conversations[start_idx:end_idx]
        encodings = tokenize(batch_convs, tokenizer).to(device)

        with activ_extractor.capture():
            _ = model(**encodings)
        
        activations = activ_extractor.get_activations()[layer_name]  # Shape: (batch_size, seq_len, hidden_size) 
        target_mask = compute_targets_mask(encodings)  # Shape: (batch_size, seq_len)
        
        if normalize:
            activations = torch.nn.functional.normalize(activations, dim=-1)
            
        activations = activations[target_mask]  # Shape: (num_valid_tokens, hidden_size)
        total_batch = activations.sum(dim=0)  # Shape: (hidden_size,)
        count_batch = activations.size(0)

        if mean_activ is None:
            mean_activ = torch.zeros_like(total_batch)

        # update running mean
        mean_activ = mean_activ * (total_count / (count_batch + total_count)) + total_batch / (count_batch + total_count)
        total_count += count_batch

    return mean_activ


def find_redundant_principle_subspace(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    layer_name: str,
    conversations: list,
    num_iters: int = 200,
    lr: float = 0.01,
    proj_weight: float = 0.1,
    batch_size: int = 32,
) -> torch.Tensor:
    """
    Returns:
        (torch.Tensor): The redundant direction found (unit norm).
    """
    device = extract_device(model)

    manipulator = ActivationManipulator(model, layer_name)
    activ_extractor = ActivationExtractor(model, layer_name)

    with torch.no_grad():
        dummy_input = tokenize([{"role": "user", "content": "Hello"}], tokenizer).to(device)
        with activ_extractor.capture():
            _ = model(**dummy_input)
        activations = activ_extractor.get_activations()[layer_name]
        layer_dim = activations.size(-1)

    if layer_dim != model.config.hidden_size:
        print(f"Warning: Layer {layer_name} dimension {layer_dim} does not match model hidden size {model.config.hidden_size}")

    w = torch.randn(layer_dim, device=device, dtype=model.dtype, requires_grad=True)
    optim = torch.optim.Adam([w], lr=lr)

    num_conversations = len(conversations)
    num_batches = (num_conversations + batch_size - 1) // batch_size

    print(f"Searching for redundant 1D subspace over {num_iters} iterations...")
    print(f"Using {num_conversations} conversations in {num_batches} batches of size {batch_size}")
    print(f"Loss balance: KL-div weight=1.0, projection weight={proj_weight}")

    best_score = -float("inf")
    best_direction = None

    def subtract_projection(activations: torch.Tensor) -> torch.Tensor:
        projection = project(activations, direction=w, normalize=True)
        return activations - projection

    mean_activ = compute_empirical_mean(
        model=model,
        tokenizer=tokenizer,
        layer_name=layer_name,
        conversations=conversations,
        num_iters=100,
        batch_size=batch_size,
        normalize=True,
    )

    for iteration in range(num_iters):
        batch_indices = torch.randperm(num_conversations)[:batch_size].tolist()
        batch_convs = [conversations[i] for i in batch_indices]
        encodings = tokenize(batch_convs, tokenizer).to(device)

        optim.zero_grad()
        v = torch.nn.functional.normalize(w, dim=0)

        with torch.no_grad(), activ_extractor.capture():
            baseline_logits = model(**encodings).logits.detach()
            activations = activ_extractor.get_activations()[layer_name].detach()

        manipulator.set_manipulation(subtract_projection)
        with manipulator.capture():
            modified_logits = model(**encodings).logits

        targets_mask = compute_targets_mask(encodings)

        # TODO: we might want to compute KL-div over top-k tokens, and not all.
        kl_div = Loss.kl_divergence(
            baseline_logits=baseline_logits,
            modified_logits=modified_logits,
            targets_mask=targets_mask,
            top_k=None,
        )

        activations_normalized = torch.nn.functional.normalize(activations, dim=-1)

        proj_var = Loss.projection_total_variance(
            activations=activations_normalized,
            direction=v,
            targets_mask=targets_mask,
            mean_activation=mean_activ,
        )

        loss = kl_div - proj_weight * proj_var
        loss.backward()
        optim.step()

        top10_agreement = Metrics.topk_agreement(
            baseline_logits=baseline_logits.detach(),
            modified_logits=modified_logits.detach(),
            targets_mask=targets_mask,
            top_k=10,
        )

        top1_accuracy = Metrics.top1_accuracy(
            baseline_logits=baseline_logits.detach(),
            modified_logits=modified_logits.detach(),
            targets_mask=targets_mask,
        )

        score_val = Metrics.redundancy_score(
            proj_norm=proj_var.item(),
            top1_acc=top1_accuracy,
            top10_agr=top10_agreement,
        )

        if score_val > best_score:
            best_score = score_val
            best_direction = v.detach().clone()

        if iteration % 20 == 0:
            print(
                f"Iter {iteration:3d}: Top-1-Acc={top1_accuracy:.4f}, "
                f"Top-10-Agr={top10_agreement:.4f}, "
                f"KL-Div={kl_div.item():.6f}, "
                f"Proj-Var={proj_var.item():.4f}, "
                f"Score={score_val:.6f}, "
                f"Loss={loss.item():.6f}"
            )

    if best_direction is None:
        best_direction = v.detach().clone()

    print(f"\n✓ Best score: {best_score:.4f}")
    print(f"✓ Final direction norm: {best_direction.norm().item():.6f}")

    return best_direction


@torch.inference_mode()
def validate_redundant_subspace(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    layer_name: str,
    convs: list[Conv],
    redundant_dir: torch.Tensor,
    batch_size: int = 8,
) -> dict[str, float]:
    device = extract_device(model)

    num_samples = len(convs)
    num_batches = (num_samples + batch_size - 1) // batch_size

    print(f"Validating over {num_samples} samples in {num_batches} batches...")

    activ_manipulator = ActivationManipulator(model, layer_name)
    activ_extractor = ActivationExtractor(model, layer_name)

    def subtract_projection(activations: torch.Tensor) -> torch.Tensor:
        projection = project(activations, direction=redundant_dir, normalize=True)
        return activations - projection

    METRICS = {
        "top1_acc": 0.0,
        "top10_agr": 0.0,
        "kl_div": 0.0,
        "proj_l2_raw": 0.0,
        "proj_l2_rel": 0.0,
        "proj_var_raw": 0.0,
        "proj_var_rel": 0.0,
        "full_l2_raw": 0.0,
        "full_var_raw": 0.0,
        "full_var_rel": 0.0,
    }

    mean_activ_normalized = compute_empirical_mean(
        model=model,
        tokenizer=tokenizer,
        layer_name=layer_name,
        conversations=convs,
        num_iters=100,
        batch_size=batch_size,
        normalize=True,
    )
    
    mean_activ = compute_empirical_mean(
        model=model,
        tokenizer=tokenizer,
        layer_name=layer_name,
        conversations=convs,
        num_iters=100,
        batch_size=batch_size,
        normalize=True,
    )
    
    for batch_idx in tqdm(range(num_batches), leave=False, desc="Validation"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)

        batch_convs = convs[start_idx:end_idx]
        encodings = tokenize(batch_convs, tokenizer).to(device)

        with activ_extractor.capture():
            baseline_logits = model(**encodings).logits
            activations = activ_extractor.get_activations()[layer_name]

        activ_manipulator.set_manipulation(subtract_projection)
        with activ_manipulator.capture():
            modified_logits = model(**encodings).logits

        targets_mask = compute_targets_mask(encodings)

        activations_normalized = torch.nn.functional.normalize(activations, dim=-1)

        METRICS["proj_l2_raw"] += Loss.projection_l2_norm(
            activations=activations,
            direction=redundant_dir,
            targets_mask=targets_mask,
            reduction="mean",
        ).item()
        
        METRICS["proj_l2_rel"] += Loss.projection_l2_norm(
            activations=activations_normalized,
            direction=redundant_dir,
            targets_mask=targets_mask,
            reduction="mean",
        ).item()
        
        METRICS["proj_var_raw"] += Loss.projection_total_variance(
            activations=activations,
            direction=redundant_dir,
            targets_mask=targets_mask,
            mean_activation=mean_activ,
        ).item()

        METRICS["proj_var_rel"] += Loss.projection_total_variance(
            activations=activations_normalized,
            direction=redundant_dir,
            targets_mask=targets_mask,
            mean_activation=mean_activ_normalized,
        ).item()
        
        METRICS["full_l2_raw"] += Loss.l2_norm(
            activations=activations,
            targets_mask=targets_mask,
            reduction="mean",
        ).item()
        
        METRICS["full_var_raw"] += Loss.total_variance(
            activations=activations,
            targets_mask=targets_mask,
            mean_activation=mean_activ,
            reduction="mean",
        ).item()
        
        METRICS["full_var_rel"] += Loss.total_variance(
            activations=activations_normalized,
            targets_mask=targets_mask,
            mean_activation=mean_activ_normalized,
            reduction="mean",
        ).item()

        METRICS["kl_div"] += Loss.kl_divergence(
            baseline_logits=baseline_logits,
            modified_logits=modified_logits,
            targets_mask=targets_mask,
        ).item()

        METRICS["top1_acc"] += Metrics.top1_accuracy(
            baseline_logits=baseline_logits,
            modified_logits=modified_logits,
            targets_mask=targets_mask,
        )

        METRICS["top10_agr"] += Metrics.topk_agreement(
            baseline_logits=baseline_logits,
            modified_logits=modified_logits,
            targets_mask=targets_mask,
            top_k=10,
        )

    METRICS = {k: v / num_batches for k, v in METRICS.items()}

    METRICS["redundancy_score"] = Metrics.redundancy_score(
        proj_norm=METRICS["proj_var_rel"],
        top1_acc=METRICS["top1_acc"],
        top10_agr=METRICS["top10_agr"],
    )

    return METRICS
