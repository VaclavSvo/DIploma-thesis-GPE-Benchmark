-- Setup --
scenario: nucleation_collision   backend: CuPy (GPU)   splitstep_order: 4
grid: N=384  (N^3=56,623,104 points)   L=16.0   dx=4.1667e-02
[PASS] velocity_resolution: k_char=3.55, k_nyquist=75.4 (need > 2.0x k_char = 7.1); OK
  scale_cutoff_to_grid: E_floor=1.239  E_target=3.152 (growth=2.54)  E_ceiling=942.6  -> effective cutoff_multiplier=7.633 (on target)

-- Ground state --
{'hbar': 1.0, 'm': 1.0, 'system': 'natural units (hbar=m=1)'}
ground state prepared in 68.27s

-- TW ensemble --
noise cutoff: 1141/56623104 modes included   temperature_natural=0.0 (vacuum-only)
n_traj=8   batch_size=1
dt=2.3576e-03   total_steps=2142   n_blocks=119

-- Nucleation detector --
planes: x,y,z   mask f=0.08 (closed 4 cells)   density dip: True
link cutoff: 1.8 dx   movie planes: x,y,z (trajectory 0)
healing length xi=1.556   dx/xi=0.0268 -- cores resolved
  TW ensemble: chunk 1/8 done (1/8 trajectories)
  TW ensemble: chunk 2/8 done (2/8 trajectories)
  TW ensemble: chunk 3/8 done (3/8 trajectories)
  TW ensemble: chunk 4/8 done (4/8 trajectories)
