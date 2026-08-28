"""IPS driver for the Proto-MPEX linear-machine example workflow.

This driver orchestrates a simple, mostly-linear pipeline via IPS ports:

1. `GRIDGEN` (mesh generation) is run once during `init(...)`.
2. Each `step(...)` runs `ELECTRON_HEATING` followed by `HERMES`.

See `examples/proto_mpex/ipsbout.config` for the wiring of ports and files.
"""

from ipsframework import Component

class hermes_linear_driver(Component):
    """Driver for a linear device workflow with electron heating + Hermes-3.

    Ports
    -----
    Expected ports (by name):
    - ``GRIDGEN``: a worker that produces a BOUT++ grid file and updates state.
    - ``ELECTRON_HEATING``: a worker that writes `Pe_src` into the grid file.
    - ``HERMES``: a Hermes/BOUT++ worker that runs the simulation.

    Notes on IPS lifecycle usage
    ----------------------------
    This driver calls each worker's `init/step/finalize` for every `step(...)`
    invocation. Some IPS workflows instead call `init(...)` once at the
    beginning and only call `step(...)` subsequently; this driver chooses the
    simpler "always call init" pattern to keep workers self-contained.
    """

    def init(self, timeStamp=0.0):
        """Run the mesh generator once and initialise the IPS state.

        The GRIDGEN component is expected to put its products into IPS state
        (typically by listing them in `STATE_FILES` and calling
        `services.update_state()`).
        """
        try:
            gridgen = self.services.get_port("GRIDGEN")
            self.services.call(gridgen, "init", timeStamp)
            self.services.call(gridgen, "step", timeStamp)
            self.services.call(gridgen, "finalize", timeStamp)
        except KeyError:
            self.services.warning(f"Port 'GRIDGEN' not found. Skipping")

        # Stage any initial state files
        # This can be used to copy an initial restart file
        self.services.update_state()

    def step(self, timeStamp=0.0):
        """Run the heating mapper and Hermes worker for this IPS step."""
        for name in ["ELECTRON_HEATING", "HERMES"]:
            try:
                component = self.services.get_port(name)
            except KeyError:
                self.services.info(f"Port '{name}' not found. Skipping")
                continue
            self.services.info(f"Running port '{name}'")
            self.services.call(component, "init", timeStamp)
            self.services.call(component, "step", timeStamp)
            self.services.call(component, "finalize", timeStamp)
