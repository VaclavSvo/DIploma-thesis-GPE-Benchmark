# Performance and memory

Measured, not estimated. Two backends behave differently enough that the
numbers are kept separate.

---

## Memory budget (GPU)

Per grid point, TW scenario at `SPLITSTEP_ORDER = 4`:

- persistent: `K_sq` 4 B + `V` 4 B + `mask` 1 B + `psi_mean` 8 B ~= 17 B
- one resident trajectory: `psi` 8 B
- `compute_diagnostics()` transient: ~20–24 B

**Measured on real hardware: ~45.5 B/point.** `tw_collision` at `N=512`,
order 4, `batch_size=1`, 8 trajectories, `T_TOTAL=5` peaked at 5821 MiB of 6144
on a 6 GB GTX 1660 Ti — it works, but at ~95% utilisation, so treat `N=512` TW
as "works for this config on this card" rather than "safe with margin".
Projections at that coefficient: `N=384` ~= 2460 MiB, `N=448` ~= 3900 MiB, both
with real headroom.

That 45.5 B/point matches the *logical* estimate almost exactly, meaning the
~1.9x CuPy pool-overhead multiplier this project used to assume no longer
applies. It was measured under the old allocation pattern (order-4 kinetic
operators persisted for the whole run, diagnostics holding all three derivative
arrays at once, `batch_size=2`) — lots of large arrays alive concurrently,
which is what drives pool fragmentation. What removed it:

- `batch_size=1`, so only one trajectory's working set is ever resident
  (chunking is exact, not an approximation — it only trades memory for
  wall-clock).
- order-4's kinetic operators built per `run_block()` call instead of
  persisted, so 16 B/point is not resident during the diagnostics peak.
- `compute_diagnostics()` streamed rather than holding all three derivatives,
  down from a 40 B/point transient. This is the function with the OOM history
  and it sits on every gate's critical path, so it has a dedicated regression
  test (`tests/test_diagnostics_memory_opt.py`) that reimplements the old
  formula and checks the new one against it.
- `ensemble.run_tw_ensemble(diagnostics_every=N)` recomputes diagnostics every
  N blocks instead of every block, keeping chunk endpoints fresh. Available,
  not wired into `config.py` — pick a cadence if you want it.

CPU-only scaling, measured the same way (peak RSS, real production pipeline):
~113 B/point at `N=48` falling to ~98 B/point by `N=128`, i.e. roughly 2x the
GPU figure and flattening with `N`.

---

## Grid size: pick small prime factors

FFT cost is dominated by `N`'s **largest prime factor**, not by `N` itself.
cuFFT and pocketfft are fast for products of 2, 3, 5, 7 (and handle 11, 13)
and fall onto a much slower general path otherwise — worst case `N` prime.

Measured on CPU (`scipy.fft`, 4 threads, complex64, per grid point):

| N | largest prime factor | ns / point |
|---|---|---|
| 240 | 5 | 8.8 |
| 248 | **31** | **11.2** |
| 250 | 5 | 8.9 |
| 252 | 7 | 9.0 |
| 256 | 2 | 11.8 * |

\* `N=256` is slower per point here only because it is a larger transform; the
point is the 248-vs-250 comparison, where a 0.8% change in `dx` buys 25% of the
FFT time. The gap is larger on GPU, where a prime factor of 31 forces cuFFT's
Bluestein algorithm.

`gpe3d/backend.check_fft_grid_size()` warns (does not abort) and names the
nearest good size. The default nucleation scenario used to run at `N=248`; it
is now `N=250 = 2 * 5^3`.

---

## Optimisation pass, September 2026

Measured on CPU at `N=96` unless stated. Correctness of every item is pinned by
a test; see [`AUDIT.md`](AUDIT.md) §C for what each one changed.

| | before | after | |
|---|---|---|---|
| `_energy()` | 69.2 ms | 14.2 ms | **4.9x** |
| `_energy()` (N=48, x20) | 264 ms | 42 ms | **6.3x** |
| `compute_diagnostics()` | 102 ms | 97 ms | 1.05x on CPU; 4 transforms instead of 6, so larger on GPU |
| `detect_frame()`, 3 plane families | 93.9 ms | 64.9 ms | **1.45x** |
| `winding_volume()` | 21.4 ms | 13.5 ms | 1.58x |
| imaginary-time ground state (N=48) | 6.75 s | 5.77 s | 1.17x |
| FFT per point, `N=248 -> 250` | 11.2 ns | 8.9 ns | **1.25x** |

What did it:

1. **Energy by Parseval.** `E_kin = (dV/N^3) sum_k (K^2/2)|psi_k|^2`, and
   `K^2 = kx^2+ky^2+kz^2` separates, so `|psi_k|^2` reduces onto three
   length-`N` marginals. One forward FFT instead of three forward plus three
   inverse, and no complex derivative array at all. On the imaginary-time
   convergence check and every gate's critical path.
2. **One shared forward FFT in `compute_diagnostics()`.** Angular momentum
   genuinely needs the real-space current, so the inverse transforms stay — but
   `psi` was being transformed forward once per axis. 6 transforms -> 4.
3. **Density slices stop copying the whole grid.** `get_density_numpy()[n]`
   built an `N^3` float32 array and moved all of it across PCIe to keep one
   plane: 226 MB per movie frame at `N=384`. Now `get_density_slice_numpy()`
   slices `psi` first. Same fix in the ensemble runner, where it was happening
   inside the diagnostics peak.
4. **Shared phase differences in the detector.** Each plane family needs the
   two axes transverse to its normal, so across three families every axis is
   needed exactly twice — and was computed twice. `arctan2` over the full grid
   is the detector's most expensive operation; it now runs 3 times per
   trajectory-frame instead of 6.
5. **Fewer full-grid temporaries.** In-place `w /= 2pi`; one shared `rint(w)`
   for the charge and the integrality check; `max(max(d), -min(d))` instead of
   `max(abs(d))`; in-place `|=`/`&=` in `dilate`/`erode` (at
   `close_passes = 4` the out-of-place form allocated 24 full boolean grids per
   call, twice per frame); in-place box filter in `local_density_dip`.
6. **An FFT-friendly production grid** (see above).

---

## Reference run

`tw_collision` at `N=384` (56.6M points), `L=16`, order 4, GTX 1660 Ti:
ground state 68 s; 1246 real-time steps in 866 s (1.4 steps/s); gates passed at
0.34% norm and energy drift; depletion 7.80%.

(Measured before the September 2026 pass. Expect the ground state to be
noticeably faster — the imaginary-time convergence check was 6x — and the
detector-heavy nucleation runs ~1.5x faster in their detection phase.)
