import torch


def precompute_freqs_cis(
    dim: int, end: int, theta: float, device: torch.device | None = None
) -> torch.Tensor:
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim)
    )
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    freqs_cis_k: torch.Tensor | None = None,
    olmo: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if freqs_cis_k is None:
        freqs_cis_k = freqs_cis.clone()
        
    if olmo:
        def to_complex(x):
            H, T, D = x.shape
            # Reshape into (B,H,T,D/2,2) as in OLMo
            x = x.float().reshape(H, T, 2, D // 2)  # <-- different from LLaMA
            x = x.transpose(-1, -2)                    # swap order of halves
            return torch.view_as_complex(x.contiguous())
        xq_ = to_complex(xq)
        xk_ = to_complex(xk)
    else:
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:, None, :]
    freqs_cis_k = freqs_cis_k[:, None, :]


    xq_out = torch.view_as_real(xq_ * freqs_cis)
    xk_out = torch.view_as_real(xk_ * freqs_cis_k)
    
    if olmo:
        xq_out = xq_out.transpose(-2,-1)
        xk_out = xk_out.transpose(-2,-1)
    return xq_out.type_as(xq).flatten(-2), xk_out.type_as(xk).flatten(-2)
