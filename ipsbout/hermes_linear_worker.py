"""
Linear machine turbulence simulations with Hermes-3/BOUT++
"""

import os

import numpy as np
import xbout
from boututils.datafile import DataFile

from .bout_worker import bout_worker

# BOUT.inp settings file for Hermes-3 turbulence simulation
options_template = """
nout = {NOUT}               # Number of output steps
timestep = {TIMESTEP}       # Output timestep, normalised ion cyclotron times [1/Omega_ci]

[output]
type = adios

[restart_files]
type = adios

[mesh]
file = "{GRIDFILE}"

calcParallelSlices_on_communicate = false
extrapolate_y = false  # Can result in negative Jacobians in guard cells

[mesh:paralleltransform]
type = identity

[solver]
mxstep = 1000000

[hermes]
components = (e, d+, d, sound_speed, vorticity,
              braginskii_collisions, 
              braginskii_heat_exchange,
              sheath_boundary,
              braginskii_conduction,
              braginskii_friction,
              braginskii_ion_viscosity,
              reactions,
              sheath_boundary_penalty,
              )

Nnorm = 1e20  # Reference density [m^-3]
Bnorm = 1   # Reference magnetic field [T]
Tnorm = 1   # Reference temperature [eV]

[vorticity]
diamagnetic = true   # Include diamagnetic current?
diamagnetic_polarisation = true # Include diamagnetic drift in polarisation current?
average_atomic_mass = `d+`:AA   # Weighted average atomic mass, for polarisaion current
poloidal_flows = false  # Include axial ExB flow
split_n0 = false  # Split phi into n=0 and n!=0 components
phi_boundary_relax = true
phi_boundary_timescale = 1e-6

[sheath_boundary_penalty]
gamma_e = {SHEATH_GAMMA_E}
gamma_i = {SHEATH_GAMMA_I}
penalty_timescale = {PENALTY_TIMESCALE} # Seconds
surface_terms = true

################################################################
# Electrons

[e]
# Evolve the electron density, parallel momentum, and fix Te
type = evolve_density, evolve_momentum, evolve_pressure

AA = 1 / 1836
charge = -1

poloidal_flows = false
diagnose = true

[Ne]
bndry_all = neumann

function = 1e-1 * exp(-x^2) # Starting density profile [x10^18 m^-3]

[Pe]
bndry_all = neumann
function = 5 * exp(-x^2) * Ne:function * (1. + 0.1 * mixmode(z - y))
# Mesh file contains Pe_src source

################################################################
# Deuterium ions
[d+]
# Set ion density from quasineutrality, evolve parallel flow
type = quasineutral, evolve_momentum, evolve_pressure

AA = 2       # Atomic mass
charge = 1

poloidal_flows = false

[Pd+]
bndry_all = neumann
function = 5 * exp(-x^2) * Ne:function

################################################################
# Deuterium atoms
[d]
# Set a fixed background of helium atoms
type = fixed_density, fixed_velocity, isothermal

AA = 2       # Atomic mass
charge = 0

density = {NEUTRAL_DENSITY}      # Density in m^-3
velocity = 0        # Parallel flow velocity in m/s
temperature = {NEUTRAL_TEMPERATURE}   # Atom temperature in eV

################################################################

[reactions]
type = (
        d + e -> d+ + 2e, # Ionisation
	    d+ + e -> d,      # Recombination
       )
"""

options_defaults = {
    "NOUT": 2,
    "TIMESTEP": 1e-3,
    "GRIDFILE": "",
    "SHEATH_GAMMA_E": 3.5,
    "SHEATH_GAMMA_I": 3.5,
    "PENALTY_TIMESCALE": 1e-4,
    "NEUTRAL_DENSITY": 1e19,
    "NEUTRAL_TEMPERATURE": 0.1,
}


class hermes_linear_worker(bout_worker):
    """Hermes-3 linear-machine turbulence worker.

    This component:

    1. Stages any configured input files.
    2. Stages the grid file (and optionally other state) into the working dir.
    3. Writes a `BOUT.inp` from `options_template`.
    4. Runs Hermes-3 via :class:`ipsbout.bout_worker.bout_worker`.
    5. Post-processes `BOUT.dmp.bp` into a compact NetCDF summary file
       (`PLASMAFILE`) containing time-averaged `Ne` and `Te`.
    6. Stages outputs and updates IPS state.

    Configuration parameters
    ------------------------
    Required (typical IPS config keys):
    - ``GRIDFILE``: BOUT++ grid file (usually provided via state).
    - ``PLASMAFILE``: output NetCDF plasma summary file.
    - ``BIN_PATH``: Hermes-3 executable path.
    - ``NPROC``: requested processor count.

    Optional:
    - ``RESTART_TARFILE``: path to a restart tarball; if present, it is used
      and updated by :meth:`ipsbout.bout_worker.bout_worker.step`.
    - ``NEUTRAL_DENSITY`` / ``NEUTRAL_TEMPERATURE``: injected into the input
      template.
    - ``NOUT`` and ``TIMESTEP`` : Number and size (normalized) of output steps
    - ``SHEATH_GAMMA_E`` and ``SHEATH_GAMMA_I`` sheath heat transmission factors
    - ``PENALTY_TIMESCALE`` in seconds

    State-file handling (deviation from common IPS patterns)
    --------------------------------------------------------
    Standard IPS usage is to put all persistent state in ``STATE_FILES`` and
    call `stage_state()` / `update_state()` with that list.

    This worker supports an additional split:
    - ``INPUT_STATE_FILES``: extra files to stage *in* for this step.
    - ``OUTPUT_STATE_FILES``: extra files to update *out* after this step.

    The union of ``STATE_FILES`` + ``INPUT_STATE_FILES`` is staged in, and the
    union of ``STATE_FILES`` + ``OUTPUT_STATE_FILES`` is updated out. This is
    used in ``examples/proto_mpex/ipsbout.config`` to read `GRIDFILE` from the
    state but only persist the restart tarball (plus optional derived products).
    """

    def __init__(self, services, config):
        """Construct the worker and initialise default Hermes options."""
        super().__init__(services, config)
        self.transport_options = options_defaults.copy()

    def step(self, timestamp=0.0):
        """Run Hermes-3 and write a condensed plasma-profile output.

        The Hermes run is executed by calling :meth:`bout_worker.step` after
        generating a `BOUT.inp` in the working directory.

        This method expects the Hermes executable to produce `BOUT.dmp.bp`
        (ADIOS2 output). The post-processing step uses `xbout` to read that
        dataset and writes `PLASMAFILE` as a NetCDF file.
        """
        self.services.info(f"Hermes linear machine worker step {timestamp}")

        self.services.stage_input_files(self.INPUT_FILES)

        input_state_files = getattr(self, 'STATE_FILES', '').split() + getattr(self, 'INPUT_STATE_FILES', '').split()
        self.services.info(f"Staging state files {input_state_files}")
        self.services.stage_state(state_files=input_state_files)

        if (not hasattr(self, "GRIDFILE")) or (self.GRIDFILE == ""):
            raise ValueError("GRIDFILE must be set to the input grid file.")
        
        if (not hasattr(self, "PLASMAFILE")) or (self.PLASMAFILE == ""):
            raise ValueError("PLASMAFILE must be set to the output plasma state file.")

        for key, default_value in self.transport_options.items():
            self.transport_options[key] = getattr(self, key, default_value)

        cwd = self.services.get_working_dir()

        options_file = os.path.join(cwd, "BOUT.inp")

        self.services.info(f"Options file : {options_file}")
        self.services.info(f"Options      : {self.transport_options}")

        with open(options_file, "wt") as f:
            f.write(options_template.format(**self.transport_options))

        self.OPTIONS_INP = options_file

        # Call bout_worker.step to run the simulation
        super().step(timestamp)

        # Extract data
        ds = xbout.open_boutdataset("BOUT.dmp.bp", info=False)

        with DataFile(self.GRIDFILE) as f:
            R = np.array(f["Rxy"][2:-2, :])
            Z = np.array(f["Zxy"][2:-2, :])

        wci = ds.metadata["Omega_ci"]
        Nnorm = ds.metadata["Nnorm"]
        Tnorm = ds.metadata["Tnorm"]
        Ne_av = np.array(ds["Ne"].mean(dim="t")[2:-2, :, :] * Nnorm)  # in m^-3
        Te_av = np.array(ds["Te"].mean(dim="t")[2:-2, :, :] * Tnorm)  # in eV
        t_array = ds["t"] / wci * 1e6  # in microseconds

        nx, ny, nz = Ne_av.shape
        angle = 2 * np.pi * np.arange(nz) / nz

        # Expand the R, Z and angle arrays to [nx, ny, nz]
        R = np.repeat(R[..., np.newaxis], nz, axis=-1)
        Z = np.repeat(Z[..., np.newaxis], nz, axis=-1)
        angle = np.repeat(
            np.repeat(angle[np.newaxis, ...], ny, axis=0)[np.newaxis, ...], nx, axis=0
        )

        with DataFile(self.PLASMAFILE, create=True) as f:
            f["r"] = R
            f["z"] = Z
            f["angle"] = angle
            f["electron_density"] = Ne_av
            f["electron_temperature"] = Te_av
            f["average_time_us"] = float(t_array[-1] - t_array[0])
            f["simulation_time_us"] = float(t_array[-1])

        self.services.info(f"Plasma state written to {self.PLASMAFILE}")

        if hasattr(self, "OUTPUT_FILES"):
            self.services.stage_output_files(timestamp, self.OUTPUT_FILES)

        output_state_files = getattr(self, 'STATE_FILES', '').split() + getattr(self, 'OUTPUT_STATE_FILES', '').split()
        self.services.info(f"Updating state files {output_state_files}")
        self.services.update_state(state_files=output_state_files)  # Update plasma state
