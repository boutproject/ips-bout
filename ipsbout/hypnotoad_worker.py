from ipsframework import Component

class hypnotoad_worker(Component):
    """Hypnotoad mesh-generator worker.

    This worker wraps Hypnotoad to generate a BOUT++ grid file from a tokamak
    equilibrium in G-EQDSK format.

    Configuration parameters
    ------------------------
    Required:
    - ``GEQDSK``: input equilibrium file in G-EQDSK format.
    - ``GRIDFILE``: output BOUT++ grid filename.

    Optional:
    - ``OPTIONS_YAML``: YAML file of Hypnotoad settings.

    Staging conventions
    -------------------
    The typical IPS pattern is:
    - ``INPUT_FILES = ${OPTIONS_YAML} ${GEQDSK}``
    - ``OUTPUT_FILES = ${GRIDFILE}``

    Notes
    -----
    Hypnotoad imports are done inside :meth:`step` so that this module can be
    imported in environments without Hypnotoad installed (e.g. documentation
    builds) as long as the component is not executed.
    """
    def __init__(self, services, config):
        """Construct the worker."""
        super().__init__(services, config)

    def init(self, timestamp=0.0, **kwargs):
        """IPS lifecycle hook called before the first step."""
        self.services.info(f"Created {self.__class__}")

    def step(self, timestamp=0.0):
        """Generate the grid file using Hypnotoad and write ``GRIDFILE``."""
        self.services.info("Importing hypnotoad modules")
        from hypnotoad.cases import tokamak
        from hypnotoad.core.mesh import BoutMesh

        self.services.info(f"INPUT_FILES: {self.INPUT_FILES}")
        self.services.info(f"OUTPUT_FILES: {self.OUTPUT_FILES}")

        if (not hasattr(self, "GEQDSK")) or (self.GEQDSK == ""):
            raise ValueError("GEQDSK must be set to the equilibrium file")

        if (not hasattr(self, "GRIDFILE")) or (self.GRIDFILE == ""):
            raise ValueError("GRIDFILE must be set to the output grid file.")

        if hasattr(self, 'OPTIONS_YAML') and self.OPTIONS_YAML != "":
            # Read a yaml options file
            self.services.info(f"Reading options from YAML file '{self.OPTIONS_YAML}'")
            import yaml
            with open(self.OPTIONS_YAML, "r") as inputfile:
                options = yaml.safe_load(inputfile)
        else:
            options = {}

        possible_options = (
            [opt for opt in tokamak.TokamakEquilibrium.user_options_factory.defaults]
            + [
                opt
                for opt in tokamak.TokamakEquilibrium.nonorthogonal_options_factory.defaults
            ]
            + [opt for opt in BoutMesh.user_options_factory.defaults]
        )

        unused_options = [opt for opt in options if opt not in possible_options]
        if unused_options != []:
            self.services.info(f"Unused options: {unused_options}")

        self.services.info(f"Reading GEQDSK file '{self.GEQDSK}'")
        with open(self.GEQDSK, "rt") as fh:
            eq = tokamak.read_geqdsk(fh, settings=options, nonorthogonal_settings=options)

        self.services.info("Create mesh")
        mesh = BoutMesh(eq, options)
        mesh.calculateRZ()
        mesh.geometry()

        self.services.info("Writing mesh")
        mesh.writeGridfile(self.GRIDFILE)
        
        self.services.info("Finished")
