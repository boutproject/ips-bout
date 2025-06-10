from ipsframework import Component
import logging

logger = logging.getLogger(__name__)


class bout_worker(Component):
    """
    # BOUT++ simulation



    ## Configuration options

    OPTIONS_INP = BOUT.inp options file
    GRIDFILE = BOUT++ grid file
    BIN_PATH = BOUT++ executable

    The following options should be set:

    INPUT_FILES = ${OPTIONS_INP} ${GRIDFILE}
    OUTPUT_FILES =

    """

    def __init__(self, services, config):
        """
        Starting a new simulation
        """
        super().__init__(services, config)
        logger.info(f"Created {self.__class__}")
        self.restarting = False

    def step(self, timestamp=0.0):
        logger.debug(f"BOUT++ step {timestamp}")

        if (not hasattr(self, "OPTIONS_INP")) or (self.OPTIONS_INP == ""):
            raise ValueError("OPTIONS_INP must be set to the input options file")

        if (not hasattr(self, "GRIDFILE")) or (self.GRIDFILE == ""):
            raise ValueError("GRIDFILE must be set to the input grid file.")

        if (not hasattr(self, "BIN_PATH")) or (self.BIN_PATH == ""):
            raise ValueError("BIN_PATH must be set to a BOUT++ executable")
        # Command line argument "-f <option file> -d <data file>"
        command_line_args = f"-f {self.OPTIONS_INP} "
        if self.restarting:
            command_line_args += " restart"
        logger.debug(f"Command line arguments: '{command_line_args}'")

        # Run simulation
        # self.services.launch_task()

        logger.debug("Finished BOUT++ step")
        self.restarting = True  # A following step will restart

    def restart(self, timestamp=0.0):
        """
        Called to restart a simulation from a previous simulation step
        """
        logger.debug("BOUT++ restart")
        self.restarting = False  # Used later when running BOUT++
        pass
