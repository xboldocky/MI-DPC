###
# Modified version of torch.gumbel_softmax
# Includes option of disabling gumbels
# Source: https://github.com/pytorch/pytorch/blob/v2.9.1/torch/nn/functional.py#L2139
###
#%%
import torch


def gumbel_softmax(
    logits: torch.Tensor,
    tau: float = 1,
    hard: bool = False,
    eps: float = 1e-10,
    dim: int = -1,
    enable_gumbels: bool = True
) -> torch.Tensor:
    
    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")

    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
        .exponential_()
        .log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + (gumbels*enable_gumbels)) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format
        ).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

def main(): # Test fn
    logits = torch.tensor([1.0, 2.0, 3.0])

    print("=== enable_gumbels=True (stochastic) ===")
    for i in range(3):
        out = gumbel_softmax(logits, enable_gumbels=True)
        print(f"Run {i+1}: {out}")

    print("\n=== enable_gumbels=False (deterministic) ===")
    for i in range(3):
        out = gumbel_softmax(logits, enable_gumbels=False)
        print(f"Run {i+1}: {out}")

    # Optional simple check:
    out1 = gumbel_softmax(logits, enable_gumbels=False)
    out2 = gumbel_softmax(logits, enable_gumbels=False)
    print("\nDeterministic check:", torch.allclose(out1, out2))


if __name__ == "__main__":
    main()
# %%
