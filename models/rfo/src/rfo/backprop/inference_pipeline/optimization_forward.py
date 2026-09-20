"""Differentiable RF3 inference used only by RFO sequence optimization.

The public RF3 confidence model disables trunk gradients in its inference
forward. RFO needs gradients through the final trunk recycle and prediction
heads, while diffusion coordinates are deliberately treated as constants.
Use the existing model components without patching RF3's normal forward.
"""

from collections import deque

import torch
from torch.utils.checkpoint import checkpoint


def optimization_forward(model, inputs, n_cycle, coordinates, skip_diffusion=False):
    if model.training:
        raise ValueError("RFO optimization requires an RF3 model in eval mode.")
    if n_cycle < 1:
        raise ValueError("n_cycle must be at least 1.")
    if not hasattr(model, "confidence_head"):
        raise ValueError("RFO requires an RF3 checkpoint with a confidence head.")

    # Match RF3's feature casting without assuming a CUDA device.
    device_type = inputs["f"]["restype"].device.type
    if torch.is_autocast_enabled(device_type):
        dtype = torch.get_autocast_dtype(device_type)
        for key in ("msa_stack", "profile", "deletion_mean", "restype", "ref_pos"):
            if key in inputs["f"]:
                inputs["f"][key] = inputs["f"][key].to(dtype)

    recycled = deque(
        model.trunk_forward_with_recycling(f=inputs["f"], n_recycles=n_cycle),
        maxlen=1,
    ).pop()
    output = {
        "early_stopped": False,
        "X_L": None,
        "distogram": model.distogram_head(recycled["Z_II"]),
        "S_I": recycled["S_I"],
        "Z_II": recycled["Z_II"],
    }
    if skip_diffusion:
        return output

    with torch.no_grad():
        sampled = model.inference_sampler.sample_diffusion_like_af3(
            f=inputs["f"],
            S_inputs_I=recycled["S_inputs_I"],
            S_trunk_I=recycled["S_I"],
            Z_trunk_II=recycled["Z_II"],
            diffusion_module=model.diffusion_module,
            diffusion_batch_size=inputs["t"].shape[0],
            coord_atom_lvl_to_be_noised=coordinates,
        )
    coords = sampled["X_L"].detach()
    confidence = {}
    for sample in coords:
        values = checkpoint(
            model.confidence_head,
            recycled["S_inputs_I"],
            recycled["S_I"],
            recycled["Z_II"],
            sample.unsqueeze(0),
            inputs["seq"],
            inputs["rep_atom_idxs"],
            frame_atom_idxs=inputs["frame_atom_idxs"],
            use_reentrant=False,
        )
        for key, value in values.items():
            confidence.setdefault(key, []).append(value)
    output.update(
        {key: sampled[key] for key in ("X_noisy_L_traj", "X_denoised_L_traj", "t_hats")}
    )
    output["X_pred_rollout_L"] = coords
    for key in ("plddt", "pae", "pde", "exp_resolved"):
        output[key] = torch.cat(confidence[f"{key}_logits"], dim=0)
    return output
