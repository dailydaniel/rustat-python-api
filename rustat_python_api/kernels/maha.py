import torch

try:
    import triton
    import triton.language as tl

    @triton.jit
    def influence_kernel(
        MU,           # (F,P,2)
        SIGMA_INV,    # (F,P,2,2)
        LOCS,         # (N,2)  – read-only
        OUT,          # (F,N)
        F, P, N,      # sizes
        BLOCK_N: tl.constexpr,      # кол-во точек на блок
    ):
        pid_f = tl.program_id(0)                    # какой frame
        offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)   # chunk точек
        mask_n = offs_n < N

        loc_ptr = LOCS + offs_n[:, None] * 2        # strides=(2,)
        x = tl.load(loc_ptr, mask=mask_n[:, None])  # (BLOCK_N,2)

        acc = tl.zeros([BLOCK_N], dtype=tl.float32)

        for p in tl.static_range(0, 32):
            is_valid = p < P

            # --- pointers ---
            mu_ptr   = MU + pid_f * P * 2 + p * 2
            s_ptr    = SIGMA_INV + pid_f * P * 4 + p * 4

            # --- scalar loads ---
            mu0 = tl.load(mu_ptr + 0, mask=is_valid)
            mu1 = tl.load(mu_ptr + 1, mask=is_valid)

            s00 = tl.load(s_ptr + 0, mask=is_valid)
            s01 = tl.load(s_ptr + 1, mask=is_valid)
            s10 = tl.load(s_ptr + 2, mask=is_valid)
            s11 = tl.load(s_ptr + 3, mask=is_valid)

            # --- grid coords (also scalar) ---
            x0 = tl.load(LOCS + offs_n * 2 + 0, mask=mask_n)   # (BLOCK_N,)
            x1 = tl.load(LOCS + offs_n * 2 + 1, mask=mask_n)

            diff0 = x0 - mu0
            diff1 = x1 - mu1

            tmp0  = diff0 * s00 + diff1 * s01
            tmp1  = diff0 * s10 + diff1 * s11
            maha  = diff0 * tmp0 + diff1 * tmp1

            acc += tl.where(is_valid, tl.exp(-0.5 * maha), 0.0)

        out_ptr = OUT + pid_f*N + offs_n
        tl.store(out_ptr, acc, mask=mask_n)

    def triton_influence(mu, sigma_inv, locs, BLOCK_N=64):
        F, P, _ = mu.shape
        N = locs.shape[0]
        out = torch.empty((F, N), device=mu.device, dtype=mu.dtype)

        grid = (F, triton.cdiv(N, BLOCK_N))
        influence_kernel[grid](
            mu, sigma_inv, locs, out,
            F, P, N,
            BLOCK_N=BLOCK_N,
            num_warps=4
        )
        return out

except ImportError:
    def triton_influence(*args, **kwargs):
        raise RuntimeError(
            "Triton not available. This function requires Linux with NVIDIA GPU. "
            "For CPU fallback, please use the .calculate_influence() method with device='cpu'. "
            "On macOS, the package installs without GPU acceleration."
        )
