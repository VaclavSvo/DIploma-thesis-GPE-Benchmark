

-- Setup --
scenario: tw_collision   backend: CuPy (GPU)   splitstep_order: 4
grid: N=384  (N^3=56,623,104 points)   L=16.0   dx=4.1667e-02
[PASS] velocity_resolution: k_char=3.55, k_nyquist=75.4 (need > 2.0x k_char = 7.1); OK
  scale_cutoff_to_grid: E_floor=2.478  E_target=6.305 (growth=2.54)  E_ceiling=942.6  -> effective cutoff_multiplier=7.633 (on target)

-- Ground state --
{'hbar': 1.0, 'm': 1.0, 'system': 'natural units (hbar=m=1)'}
ground state prepared in 68.08s

-- TW ensemble --
noise cutoff: 3119/56623104 modes included   temperature_natural=0.0 (vacuum-only)
n_traj=4   batch_size=1
dt=2.3576e-03   total_steps=1246   n_blocks=89
  TW ensemble: chunk 1/4 done (1/4 trajectories)
  TW ensemble: chunk 2/4 done (2/4 trajectories)
  TW ensemble: chunk 3/4 done (3/4 trajectories)
  TW ensemble: chunk 4/4 done (4/4 trajectories)

-- Evolution --
real-time evolution: 1246 steps in 865.57s (1.4 steps/s)

-- Gates --
[PASS] norm: max rel drift 0.337% (tol 1%, ref=21598.7)
[PASS] energy: max rel drift 0.346% (tol 1%, ref=139050)
ALL GATES PASSED

-- TW diagnostics --
[PASS] depletion_fraction: 7.80% (n_phys=19966.4 vs n_particles_target=20000) -- OK

-- Output --
saved outputs\density_slice.gif  (90 frames @ 30fps = 3.00s, sim T_TOTAL=3.00s)