"""
2D (drift plane) turbulence simulations with Hermes-3/BOUT++
"""

from .bout_worker import bout_worker
import os

# BOUT.inp settings file
options_template = """
nout = {nout}               # Number of output steps
timestep = {timestep}       # Output timestep, normalised ion cyclotron times [1/Omega_ci]

MYG = 0  # No guard cells in Y, 2D simulation

[mesh]
nx = 260
ny = 1
nz = 256

Lrad = {Lrad}  # Radial width of domain [m]
Lpol = {Lpol}  # Poloidal size of domain [m]

Bpxy = {Bmag}  # Poloidal magnetic field [T]
Rxy = {Rmaj}   # Major radius [meters]

dx = Lrad * Rxy * Bpxy / (nx - 4)  # Poloidal flux
dz = Lpol / Rxy / nz   # Angle

hthe = 1
sinty = 0
Bxy = Bpxy
Btxy = 0
bxcvz = 1./Rxy^2  # Curvature

[mesh:paralleltransform]
type = identity

[solver]
mxstep = 10000

[hermes]
# Two species, electrons and ions
components = e, h+, vorticity, sheath_closure

recalculate_metric = true  # Calculate metrics from Rxy, Bpxy etc.

Nnorm = 1e19
Bnorm = mesh:Bxy
Tnorm = {Tsource}

[e]
type = evolve_density, evolve_pressure

charge = -1
AA = 1/1836

poloidal_flows = false  # Y flows due to ExB
thermal_conduction = false  # Parallel heat conduction

[Ne]
function = 1.0 + 1e-3*mixmode(x)*mixmode(z)

x0 = 0.2
width = 0.02
source = {Nsource} * exp(-((x-x0)/width)^2) # Units Nnorm/s

bndry_all = neumann

[Pe]
function = Ne:function

source = Ne:source * hermes:Tnorm * 1.602176634e-19

bndry_all = neumann

[h+]
# Set the density so that the plasma is quasineutral
type = quasineutral, evolve_pressure

charge = 1
AA = 1

bndry_flux = true
poloidal_flows = false
thermal_conduction = false

[Nh+]
function = Ne:function

[Ph+]
function = `Nh+:function`

source = Ne:source * hermes:Tnorm * 1.602176634e-19

bndry_all = neumann

[vorticity]

diamagnetic = true          # Include diamagnetic current?
diamagnetic_polarisation = true # Include diamagnetic drift in polarisation current?
average_atomic_mass = 1.0   # Weighted average atomic mass, for polarisaion current
bndry_flux = false          # Allow flows through radial boundaries
poloidal_flows = false      # Include poloidal ExB flow
split_n0 = false            # Split phi into n=0 and n!=0 components
phi_dissipation = false

[sheath_closure]
connection_length = {Lpar}  # meters
potential_offset = 0.0      # Potential at which sheath current is zero
sinks = true

"""

options_defaults = {
    "nout": 10,
    "timestep": 100,
    "Lpar": 50,
    "Rmaj": 1.5,
    "Bmag": 1.0,
    "Tsource": 50,
    "Lrad": 0.3,
    "Lpol": 0.3,
    "Nsource": 1e23,
}


class hermes_2Dturb_worker(bout_worker):
    """
    Hermes-3 2D (drift-plane) turbulence simulation.
    Inherits from `bout_worker` so that its functionality for running BOUT++
    simulations can be reused.
    """

    def __init__(self, services, config):
        super().__init__(services, config)
        self.options = options_defaults.copy()

    def step(self, timestamp=0.0):
        """

        # Inputs

        self.options   : dict of settings

        # Calling BOUT++

        To run BOUT++, set the following

        self.restarting       Bool   : True if restarting from previous solution
        self.OPTIONS_INP      String : Path to BOUT.inp options file

        and then call super().step(timestamp)

        """
        self.services.info(f"Hermes 2D (drift plane) turbulence worker step {timestamp}")

        cwd = self.services.get_working_dir()

        options_file = os.path.join(cwd, "BOUT.inp")

        self.services.info(f"Options file : {options_file}")
        self.services.info(f"Options      : {self.options}")

        with open(options_file, "wt") as f:
            f.write(options_template.format(**self.options))

        self.OPTIONS_INP = options_file

        # Call bout_worker.step to run the simulation
        super().step(timestamp)
