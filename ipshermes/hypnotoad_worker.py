from ipsframework import Component
import logging
logger = logging.getLogger(__name__)

class hypnotoad_worker(Component):
    """
    # Hypnotoad mesh generator

    Given a GEQDSK file and configuration settings, generate a BOUT++
    mesh that can be used in Hermes-3 simulations.

    ## Configuration options

    GEQDSK = File containing equilibrium in G-EQDSK format
    OPTIONS_YAML = File containing mesh generation options in YAML format
    GRIDFILE = BOUT++ grid file to be created

    The following options should be set:

    INPUT_FILES = ${OPTIONS_YAML} ${GEQDSK}
    OUTPUT_FILES = ${GRIDFILE}

    """
    def __init__(self, services, config):
        super().__init__(services, config)
        logger.info(f'Created {self.__class__}')

    def step(self, timestamp=0.0):
        logger.debug('Importing hypnotoad modules')
        from hypnotoad.cases import tokamak
        from hypnotoad.core.mesh import BoutMesh

        logger.debug(f'INPUT_FILES: {self.INPUT_FILES}')
        logger.debug(f'OUTPUT_FILES: {self.OUTPUT_FILES}')

        if (not hasattr(self, "GEQDSK")) or (self.GEQDSK == ""):
            raise ValueError("GEQDSK must be set to the equilibrium file")

        if (not hasattr(self, "GRIDFILE")) or (self.GRIDFILE == ""):
            raise ValueError("GRIDFILE must be set to the output grid file.")

        if hasattr(self, 'OPTIONS_YAML') and self.OPTIONS_YAML != "":
            # Read a yaml options file
            logger.debug(f"Reading options from YAML file '{self.OPTIONS_YAML}'")
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
            logger.warning(f"Unused options: {unused_options}")

        logger.debug(f"Reading GEQDSK file '{self.GEQDSK}'")
        with open(self.GEQDSK, "rt") as fh:
            eq = tokamak.read_geqdsk(fh, settings=options, nonorthogonal_settings=options)

        logger.debug("Create mesh")
        mesh = BoutMesh(eq, options)
        mesh.calculateRZ()
        mesh.geometry()

        logger.debug("Writing mesh")
        mesh.writeGridfile(self.GRIDFILE)
        
        logger.debug('Finished')
