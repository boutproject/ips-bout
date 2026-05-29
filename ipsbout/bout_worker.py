from ipsframework import Component
import logging
import tarfile
from pathlib import Path

logger = logging.getLogger(__name__)

from boututils.datafile import DataFile


class bout_worker(Component):
    """
    # BOUT++ simulation

    This class can run standalone BOUT++ simulations, or can be used
    as a base class for specific BOUT++ models.

    ## Configuration options

    OPTIONS_INP = BOUT.inp options file
    GRIDFILE = BOUT++ grid file
    BIN_PATH = BOUT++ executable

    ## Optional inputs
    RESTART_TARFILE = BOUT++ restart tarfile
    """

    def __init__(self, services, config):
        """
        Starting a new simulation
        """
        super().__init__(services, config)
        logger.info(f"Created {self.__class__}")

    def step(self, timestamp=0.0):
        """
        Run a BOUT++ simulation

        This will use at most self.NPROC processors. Fewer processors
        will be used if necessary to decompose the grid file equally
        between processors.
        """
        logger.info(f"BOUT++ step {timestamp}")

        if (not hasattr(self, "OPTIONS_INP")) or (self.OPTIONS_INP == ""):
            raise ValueError("OPTIONS_INP must be set to the input options file path")

        if (not hasattr(self, "GRIDFILE")) or (self.GRIDFILE == ""):
            raise ValueError("GRIDFILE must be set to the input grid file path.")

        if (not hasattr(self, "BIN_PATH")) or (self.BIN_PATH == ""):
            raise ValueError("BIN_PATH must be set to a BOUT++ executable path")

        if (not hasattr(self, "NPROC")) or (self.NPROC == ""):
            raise ValueError("NPROC must be set to a number of processors")

        restarting = hasattr(self, "RESTART_TARFILE") and (self.RESTART_TARFILE != "")
        if restarting:
            logger.info(f"Extracting restart tarfile '{self.RESTART_TARFILE}'")
            with tarfile.open(self.RESTART_TARFILE, "r:*") as tar:
                tar.extractall()

        logger.info(f"Running executable  : {self.BIN_PATH}")

        cwd = self.services.get_working_dir()
        logger.info(f"Working directory   : {cwd}")

        ncpu = self.num_processors(int(self.NPROC))
        logger.info(f"Number of processors: {ncpu}")

        # Command line argument "-f <option file> -d <data file>"
        command_line_args = f"-f {self.OPTIONS_INP} -d {cwd}"
        if restarting:
            command_line_args += " restart"

        # Run simulation
        logger.info(f"Arguments           : {command_line_args}")
        task_id = self.services.launch_task(
            ncpu, cwd, self.BIN_PATH, command_line_args, logfile="bout.log"
        )
        retcode = self.services.wait_task(task_id)

        # Tar restart file
        restart_files = list(Path.cwd().glob("BOUT.restart*.nc")) + list(Path.cwd().glob("BOUT.restart.bp"))
        logger.info(f"Saving restart files {restart_files}")

        restart_tarfile = self.RESTART_TARFILE if restarting else "BOUT.restart.tar"
        with tarfile.open(restart_tarfile, "w") as tar:
            for file in restart_files:
                # arcname ensures the file doesn't store the full absolute path
                tar.add(file, arcname=file.name)
        logger.info(f"Restarts saved to tarfile '{restart_tarfile}'")

        # Next step use this tarfile
        # The content of the working directory persist between `step`
        # calls so this restart file will be present next call.
        self.RESTART_TARFILE = restart_tarfile

        logger.debug("Finished BOUT++ step")

    def num_processors(self, max_nprocs: int) -> int:
        """Return the number of processors to be used, that is
        lower than or equal to `max_nprocs`.

        This will open `self.GRIDFILE` to identify how many processors
        can be used
        """
        MXG = 2  # Number of X guard cells
        MYG = 2
        with DataFile(self.GRIDFILE) as g:
            nx = g["nx"]
            ny = g["ny"]
            jyseps1_1 = g.get("jyseps1_1", -1)
            jyseps1_2 = g.get("jyseps1_2", ny // 2)
            jyseps2_1 = g.get("jyseps2_1", jyseps1_2)
            jyseps2_2 = g.get("jyseps2_2", ny - 1)
            ny_inner = g.get("ny_inner", jyseps2_1)

        MX = nx - 2 * MXG  # Number of points in X on each processor

        # Check inputs.
        # This follows BOUT++ BoutMesh
        # https://github.com/boutproject/BOUT-dev/blob/master/src/mesh/impls/bout/boutmesh.cxx#L115
        if jyseps1_1 < -1:
            jyseps1_1 = -1
        if jyseps2_1 < jyseps1_1:
            jyseps2_1 = jyseps1_1 + 1
        if jyseps1_2 < jyseps2_1:
            jyseps1_2 = jyseps2_1
        if jyseps2_2 >= ny:
            jyseps2_2 = ny - 1
        if jyseps2_2 < jyseps1_2:
            jyseps2_2 = jyseps1_2

        def valid_split(nxpe, nprocs: int) -> bool:
            """
            Checks if a processor split is valid
            """
            if nprocs % nxpe != 0:
                return False  # NXPE must be a factor of NPROCS
            if MX % nxpe != 0:
                return False  # X mesh must divide equally among NXPE processors
            nype = nprocs // nxpe
            if ny % nype != 0:
                return False  # Y mesh must divide equally among NYPE processors
            num_local_y_points = ny // nype  # Number of points in Y on each processor

            # These checks are from
            # https://github.com/boutproject/BOUT-dev/blob/master/src/mesh/impls/bout/boutmesh.cxx#L165
            if num_local_y_points < MYG and nype != 1:
                return False
            if (jyseps1_1 + 1) % num_local_y_points != 0:
                return False
            if jyseps2_1 != jyseps1_2:
                # Double null
                if (jyseps2_1 - jyseps1_1) % num_local_y_points != 0:
                    return False
                if (jyseps2_2 - jyseps1_2) % num_local_y_points != 0:
                    return False
                if (ny_inner - jyseps2_1 - 1) % num_local_y_points != 0:
                    return False
                if (jyseps1_2 - ny_inner + 1) % num_local_y_points != 0:
                    return False
            else:
                # Single null
                if (jyseps2_2 - jyseps1_1) % num_local_y_points != 0:
                    return False
            if (ny - jyseps2_2 - 1) % num_local_y_points != 0:
                return False
            return True

        def valid_nprocs(nprocs: int) -> bool:
            """
            Returns True if the number of processors can be used to
            run a BOUT++ simulation.
            """
            for nxpe in range(1, nprocs):
                if valid_split(nxpe, nprocs):
                    return True
            return False  # No valid combinations

        nprocs = max_nprocs
        while True:
            if nprocs <= 0:
                raise ValueError(f"Invalid number of processors: {max_nprocs}")
            if valid_nprocs(nprocs):
                return nprocs
            nprocs -= 1
