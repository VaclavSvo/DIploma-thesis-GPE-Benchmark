"""Array backend: CuPy when available, NumPy otherwise, plus the project's
single FFT entry points. Every other module imports `xp`/`fftn`/`ifftn` from
here rather than numpy/cupy directly, so switching backend never touches
physics code.
"""
try:
    import cupy as xp  # type: ignore
    HAS_GPU = True
except ImportError:
    import numpy as xp  # type: ignore
    HAS_GPU = False

if HAS_GPU:
    # Both are already cupy's documented defaults; set explicitly so a future
    # release changing them can't silently regress this.
    try:
        # Plan a genuine ND transform for fftn/ifftn instead of a sequence of
        # 1D ones -- matters for this project's (N,N,N) transforms.
        xp.fft.config.enable_nd_planning = True
    except AttributeError:
        pass
    try:
        # Page-locked host memory -> DMA instead of a staged copy whenever a
        # frame is pulled off the GPU (get_density_numpy, once per gif frame).
        _pinned_pool = xp.cuda.PinnedMemoryPool()
        xp.cuda.set_pinned_memory_allocator(_pinned_pool.malloc)
    except AttributeError:
        pass


def to_numpy(a):
    """Pull an array off the GPU (no-op on the CPU backend)."""
    return xp.asnumpy(a) if HAS_GPU else a


# GPU: straight to cuFFT. cupy caches plans per shape/dtype/axes, and this
# project only ever transforms one (N,N,N) complex64 shape per engine, so that
# plan is reused for the engine's whole lifetime -- explicit plan management
# would only save a dict lookup against a millisecond-scale transform.
#
# CPU: scipy.fft rather than numpy.fft, which multithreads the transform while
# preserving complex64 exactly (verified -- older numpy releases silently
# upcast FFT output to complex128, which would make this a precision change
# and not just a speed one).
if HAS_GPU:
    fftn = xp.fft.fftn
    ifftn = xp.fft.ifftn
else:
    import os as _os
    import scipy.fft as _scipy_fft

    _CPU_FFT_WORKERS = _os.cpu_count() or 1

    def fftn(a, axes=None):
        return _scipy_fft.fftn(a, axes=axes, workers=_CPU_FFT_WORKERS)

    def ifftn(a, axes=None):
        return _scipy_fft.ifftn(a, axes=axes, workers=_CPU_FFT_WORKERS)


def _largest_prime_factor(n: int) -> int:
    largest, d = 1, 2
    while d * d <= n:
        while n % d == 0:
            largest, n = d, n // d
        d += 1
    return n if n > 1 else largest


def check_fft_grid_size(N: int, max_prime: int = 13) -> None:
    """Warn if N has a large prime factor. FFT cost is dominated by N's
    largest prime factor: cuFFT/pocketfft are fast for products of small
    primes and fall off a cliff otherwise (worst case N itself prime). A bad N
    still gives the right physics, just slowly -- so a warning, not a gate.
    """
    factor = _largest_prime_factor(N)
    if factor <= max_prime:
        return
    suggestion = N
    while _largest_prime_factor(suggestion) > max_prime:
        suggestion += 1
    print(f"[perf warning] N={N} has largest prime factor {factor} -- FFTs will be "
          f"noticeably slower than a nearby FFT-friendly size (only prime factors "
          f"<= {max_prime}), e.g. N={suggestion}.")
