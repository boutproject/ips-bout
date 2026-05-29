# IPS-BOUT (`ipsbout`)

Python package providing IPS components and drivers for running BOUT++-style
executables (notably Hermes-3), plus common pre/post-processing steps used in
workflows.

This repository is primarily intended for IPS simulations where:

- a mesh/grid is generated or staged into IPS state,
- optional sources (e.g. heating) are mapped onto the grid file,
- Hermes-3 (or another BOUT++ executable) is launched via IPS services,
- derived outputs are staged to results and/or written back into IPS state.

## What’s in here

- `ipsbout/`: the installable Python package.
  - `bout_worker.py`: base worker that encapsulates “run a BOUT++ executable”
    mechanics (task launch, restart tarball handling, and processor selection).
  - `hermes_*_worker.py`: example Hermes-3 workers that generate `BOUT.inp` and
    then call `bout_worker.step(...)`.
  - `linear_mesh_generator.py`: a `GRIDGEN` worker for a linear device that
    reads an IMAS-style machine JSON, generates a BOUT++ grid, and adds a wall
    penalty mask.
  - `electron_heating_xyz.py`: worker that interpolates a Cartesian heating
    profile onto a cylindrical/toroidal BOUT++ mesh and writes `Pe_src`.
  - `hypnotoad_worker.py`: wrapper worker that runs Hypnotoad to generate a
    tokamak mesh from a G-EQDSK equilibrium.
- `examples/`: runnable IPS configs demonstrating how the components are used.
  - `examples/proto_mpex/`: end-to-end workflow (gridgen → heating → Hermes).

## Design notes (IPS-specific)

Some choices are deliberate and may differ from “standard” IPS component style:

- **Logging via IPS services**: components log via `self.services.info(...)`
  rather than module-level loggers. IPS services should not be called from
  `__init__`; any “created” logging happens in `init(...)`.

- **Workers inheriting from `bout_worker`**: Hermes/BOUT++ workers subclass
  `ipsbout.bout_worker.bout_worker` to share consistent launch behavior and
  restart handling.

- **Optional split state file lists**: some workers support `INPUT_STATE_FILES`
  and `OUTPUT_STATE_FILES` (in addition to `STATE_FILES`) to distinguish “stage
  in” vs “update out” state for a given step. See
  `examples/proto_mpex/ipsbout.config`.

## Installation

### 1) Build/install Hermes-3

Hermes-3 is included as a git submodule under `hermes-3/`.

- Build via the helper script: `./cmake-build-hermes.sh`
- Or follow the upstream Hermes-3 installation instructions:
  https://hermes3.readthedocs.io/en/latest/installation.html

For use with these workflows, Hermes-3 is typically built with ADIOS2 enabled
so it can write/read `BOUT.dmp.bp` and restart files via ADIOS2.

### 2) Install the Python package

From the repo root:

- `pip install -e .`

Dependencies are declared in `pyproject.toml`. Some components (e.g. Hypnotoad,
SciPy, xbout) are only required if you execute those specific workflows.

## Running the examples

- Proto-MPEX linear-device example: `examples/proto_mpex/`
  - Run: `cd examples/proto_mpex && ./run.sh`
  - Docs: `examples/proto_mpex/README.md`

## Testing

- `python3 -m pytest -q`
