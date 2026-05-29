# Proto-MPEX IPS example (`examples/proto_mpex`)

This folder contains a runnable IPS example workflow that couples:

- `GRIDGEN`: generate a BOUT++ grid (`proto_mpex.nc`) from a machine JSON file
- `ELECTRON_HEATING`: map a Cartesian heating profile onto that grid (writes `Pe_src`)
- `HERMES`: run Hermes-3 and write a condensed output file (`plasma_profiles.nc`)

The workflow wiring is defined in `examples/proto_mpex/ipsbout.config`.

## Quick start

From this directory:

- Run the workflow: `./run.sh`
- Or directly: `ips.py --config=ipsbout.config --platform=platform.conf`

## Inputs

- `ProtoMPEX_G.json`: machine description (IMAS-style `pf_active` + `wall`)
- `helicon_profile.txt`: heating profile as whitespace-delimited `x y z Q` (W/m³)
- `BOUT.restart.tar`: initial restart state for Hermes/BOUT++ (optional but used here)
- `hermes-3`: symlink to the Hermes-3 executable (see `BIN_PATH` in config)

## Outputs

- `plasma_profiles.nc`: compact NetCDF output produced by `ipsbout.hermes_linear_worker`
  containing time-averaged `electron_density` and `electron_temperature`.
- `work/state/`: IPS state directory. In this example it is used to pass:
  - `proto_mpex.nc` (grid + `Pe_src`)
  - `BOUT.restart.tar` (restart tarball updated after each Hermes run)

IPS also writes run logs and staging artifacts under `work/` and
`simulation_results/` depending on platform settings.

## Notes on component design

- **Logging**: components log via `self.services.info(...)` (IPS logger). IPS
  services are not used in `__init__`; any "created" logging happens in `init(...)`.

- **Workers inheriting from `bout_worker`**: the Hermes workers share common
  task-launching logic, restart tarball handling, and processor selection via
  `ipsbout.bout_worker.bout_worker`.

- **Split state file lists**: `ipsbout.hermes_linear_worker` supports
  `INPUT_STATE_FILES` and `OUTPUT_STATE_FILES` in addition to `STATE_FILES`.
  In `ipsbout.config` this is used so the worker:
  - stages `proto_mpex.nc` in from state (input)
  - updates `plasma_profiles.nc` to state (derived output)
  - persists `BOUT.restart.tar` as the main state artifact across steps

If you adapt this workflow, keep `proto_mpex.nc` listed in `STATE_FILES` for
the components that modify it in-place (e.g. `electron_heating_xyz`), so IPS
stages the updated grid forward to downstream components.

