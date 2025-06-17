"""
Transport simulations with Hermes-3/BOUT++
"""

from .bout_worker import bout_worker
import os
import logging

logger = logging.getLogger(__name__)

# BOUT.inp settings file for Hermes-3 transport simulation
options_template = """
nout = {nout}               # Number of output steps
timestep = {timestep}       # Output timestep, normalised ion cyclotron times [1/Omega_ci]

MZ = 1

# Scale density and temperature

scale_T = {scale_T}
scale_N = {scale_N}

# Core boundary condition

core_te = {core_te} * scale_T / hermes:Tnorm   # [eV]
core_ti = {core_ti} * scale_T / hermes:Tnorm   # [eV]
core_ne = {core_ne} * scale_N / hermes:Nnorm    # Nm-3

anomalous_D_e_core = {D_core}
anomalous_Chi_e_core = {Chi_e_core}
anomalous_D_e_sol = {D_core}
anomalous_Chi_e_sol = {Chi_e_sol}

anomalous_D_i_core = anomalous_D_e_core    # Same as electrons
anomalous_Chi_i_core = {Chi_i_core}
anomalous_D_i_sol = anomalous_D_e_sol    # Same as electrons
anomalous_Chi_i_sol = {Chi_i_sol}

initial_te = core_te
initial_ti = core_ti
initial_ne = core_ne

initial_pe = initial_te * initial_ne
initial_pi = initial_ti * initial_ne

# Derived quantities
core_pi = core_ne * core_ti
core_pe = core_ne * core_te

[input]
error_on_unused_options=false

[mesh]

file = "{gridfile}"

calcParallelSlices_on_communicate = false
extrapolate_y = false

# Geometry calculations for profiles
# This is adapted from https://bout-dev.readthedocs.io/en/latest/user_docs/variable_init.html#id4
# The index (i) is defined as excluding guards, hence I add "+ MXG" at the enabled
# Remember to manually change these numbers if you change grid!!!

xsep_inner = 7  # Last cell before separatrix (ixseps - 1)
nx = 20
MXG = 2
x_index = x*(nx - 2 * MXG) - 0.5 + MXG

[mesh:paralleltransform]
type = shifted

[solver]
type = cvode
mxstep = 1e6

[hermes]
components = (d+, d, e,
              collisions, sheath_boundary_simple, recycling,
              sound_speed,
              reactions,
              electron_force_balance)

Nnorm = scale_N * 1e20    # Reference density [m^-3]
Bnorm = 1                 # Reference magnetic field [T]
Tnorm = scale_T * 100     # Reference temperature [eV]

################################################################
# Ions

[d+]
type = evolve_density, evolve_momentum, evolve_pressure, anomalous_diffusion

AA = 2
charge = 1
# Profiles. Step function with a value inside separatrix (first term) and outside separatrix (second term)
anomalous_D = H((mesh:xsep_inner+1) - mesh:x_index)*anomalous_D_i_core + H(mesh:x_index - mesh:xsep_inner)*anomalous_D_i_sol
anomalous_chi = H((mesh:xsep_inner+1) - mesh:x_index)*anomalous_Chi_i_core + H(mesh:x_index - mesh:xsep_inner)*anomalous_Chi_i_sol

thermal_conduction = true

recycle_as = d
target_recycle = true
target_recycle_multiplier = {target_recycle_multiplier}  # Recycling fraction

diagnose = true

[Nd+]

function = initial_ne
bndry_core = dirichlet(core_ne)
bndry_all = neumann   # All boundaries neumann

[Pd+]

function = initial_pi
bndry_core = dirichlet(core_pi)
bndry_all = neumann   # All boundaries neumann

################################################################
# Neutrals

[d]
type = neutral_mixed

AA = 2

diagnose = true
precondition = false

################################################################
# Electrons

[e]
# Set electron density from quasineutrality,
# and parallel flow from ion flow, assuming no currents
type = quasineutral, evolve_pressure, zero_current, anomalous_diffusion

AA = 1/1836
charge = -1

anomalous_D = H((mesh:xsep_inner+1) - mesh:x_index)*anomalous_D_e_core + H(mesh:x_index - mesh:xsep_inner)*anomalous_D_e_sol
anomalous_chi = H((mesh:xsep_inner+1) - mesh:x_index)*anomalous_Chi_e_core + H(mesh:x_index - mesh:xsep_inner)*anomalous_Chi_e_sol
thermal_conduction = true

diagnose = true

[Pe]
function = initial_pe
bndry_core = dirichlet(core_pe)
bndry_all = neumann   # All other boundaries low density

################################################################

[recycling]

species = d+

[reactions]

diagnose = true

type = (
        d + e -> d+ + 2e,     # Deuterium ionisation
        d+ + e -> d,          # Deuterium recombination
        d + d+ -> d+ + d,     # Charge exchange
       )

"""

options_defaults = {
    "nout": 10,
    "timestep": 100,
    "gridfile": "",
    "scale_T": 0.01,
    "scale_N": 0.1,
    "core_te": 2000,
    "core_ti": 2000,
    "core_ne": 1e20,
    "D_core": 0.3,
    "D_sol": 0.3,
    "Chi_e_core": 0.5,
    "Chi_e_sol": 1.0,
    "Chi_i_core": 0.5,
    "Chi_i_sol": 1.0,
    "target_recycle_multiplier": 0.995,
}


class hermes_transport_worker(bout_worker):
    """
    Hermes-3 transport simulation.
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

        # Calling BOUT++

        To run BOUT++, set the following

        self.restarting       Bool   : True if restarting from previous solution
        self.OPTIONS_INP      String : Path to BOUT.inp options file

        and then call super().step(timestamp)

        """
        logger.info(f"Hermes transport worker step {timestamp}")

        if (not hasattr(self, "GRIDFILE")) or (self.GRIDFILE == ""):
            raise ValueError("GRIDFILE must be set to the input grid file.")

        self.transport_options["gridfile"] = self.GRIDFILE

        cwd = self.services.get_working_dir()

        options_file = os.path.join(cwd, "BOUT.inp")

        logger.info(f"Options file : {options_file}")
        logger.info(f"Options      : {self.transport_options}")

        with open(options_file, "wt") as f:
            f.write(options_template.format(**self.transport_options))

        self.OPTIONS_INP = options_file

        # Call bout_worker.step to run the simulation
        super().step(timestamp)
