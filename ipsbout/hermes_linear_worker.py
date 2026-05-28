"""
Linear machine turbulence simulations with Hermes-3/BOUT++
"""

import logging
import os

import numpy as np
import xbout
from boututils.datafile import DataFile

from .bout_worker import bout_worker

logger = logging.getLogger(__name__)

# BOUT.inp settings file for Hermes-3 turbulence simulation
options_template = """
nout = {nout}               # Number of output steps
timestep = {timestep}       # Output timestep, normalised ion cyclotron times [1/Omega_ci]

[output]
type = adios

[restart_files]
type = adios

[mesh]
file = "{gridfile}"

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
gamma_e = {sheath_gamma_e}
gamma_i = {sheath_gamma_i}
penalty_timescale = {penalty_timescale} # Seconds
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

density = {neutral_density}      # Density in m^-3
velocity = 0        # Parallel flow velocity in m/s
temperature = {neutral_temperature}   # Atom temperature in eV

################################################################

[reactions]
type = (
        d + e -> d+ + 2e, # Ionisation
	    d+ + e -> d,      # Recombination
       )
"""

options_defaults = {
    "nout": 400,
    "timestep": 100,
    "gridfile": "",
    "sheath_gamma_e": 3.5,
    "sheath_gamma_i": 3.5,
    "penalty_timescale": 1e-4,
    "neutral_density": 1e19,
    "neutral_temperature": 0.1,
}


class hermes_linear_worker(bout_worker):
    """
    Hermes-3 linear machine turbulence simulation.
    Inherits from `bout_worker` so that its functionality for running BOUT++
    simulations can be reused.
    """

    def __init__(self, services, config):
        super().__init__(services, config)
        logger.info(f"Created {self.__class__}")
        self.transport_options = options_defaults.copy()

    def step(self, timestamp=0.0):
        """

        # Inputs

        GRIDFILE              String : Path to the grid file
        PLASMAFILE            String : Output plasma state
        NEUTRAL_DENSITY       Float : Number density in m^-3
        NEUTRAL_TEMPERATURE   Float : Temperature in eV

        # Calling BOUT++

        To run BOUT++, set the following

        self.restarting       Bool   : True if restarting from previous solution
        self.OPTIONS_INP      String : Path to BOUT.inp options file

        and then call super().step(timestamp)

        """
        logger.info(f"Hermes linear machine worker step {timestamp}")

        self.services.stage_input_files(self.INPUT_FILES)
        self.services.stage_state()  # Fetch GRIDFILE

        if (not hasattr(self, "GRIDFILE")) or (self.GRIDFILE == ""):
            raise ValueError("GRIDFILE must be set to the input grid file.")
        self.transport_options["gridfile"] = self.GRIDFILE

        if (not hasattr(self, "PLASMAFILE")) or (self.PLASMAFILE == ""):
            raise ValueError("PLASMAFILE must be set to the output plasma state file.")

        if (not hasattr(self, "NEUTRAL_DENSITY")) or (self.NEUTRAL_DENSITY == ""):
            raise ValueError(
                "NEUTRAL_DENSITY must be set to the neutral atom density in m^-3"
            )
        self.transport_options["neutral_density"] = self.NEUTRAL_DENSITY

        if (not hasattr(self, "NEUTRAL_TEMPERATURE")) or (
            self.NEUTRAL_TEMPERATURE == ""
        ):
            raise ValueError(
                "NEUTRAL_TEMPERATURE must be set to the neutral atom temperature in eV"
            )
        self.transport_options["neutral_temperature"] = self.NEUTRAL_TEMPERATURE

        cwd = self.services.get_working_dir()

        options_file = os.path.join(cwd, "BOUT.inp")

        logger.info(f"Options file : {options_file}")
        logger.info(f"Options      : {self.transport_options}")

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

        logger.info(f"Plasma state written to {self.PLASMAFILE}")

        self.services.stage_output_files(timestamp, self.OUTPUT_FILES)
        self.services.update_state()  # Update state files
