"""IPS base worker for running BOUT++-style executables.

`bout_worker` is intended to be subclassed by IPS worker components that
generate a BOUT++ input file (typically `BOUT.inp`) and then run a BOUT++
executable (e.g. Hermes-3) via `ServicesProxy.launch_task`.

This class centralises a few implementation decisions:

- **Common launch mechanics**: build the `-f <options> -d <cwd>` command line
  and run via IPS services.
- **Restart handling**: accept a restart tarball (`RESTART_TARFILE`), extract
  it before launching, and tar up any `BOUT.restart*` outputs afterwards.
- **Processor selection**: reduce `NPROC` to a valid decomposition for the
  provided `GRIDFILE` by inspecting mesh metadata.

Subclasses are responsible for:

- ensuring required input files are present in the working directory
  (usually via `stage_input_files(...)` / `stage_state(...)`);
- writing the options file and setting `self.OPTIONS_INP`;
- setting `self.GRIDFILE`, `self.BIN_PATH`, and `self.NPROC`.
"""

import tarfile
from pathlib import Path

from boututils.datafile import DataFile
from ipsframework import Component


class bout_worker(Component):
    """Base class for BOUT++-based IPS worker components.

    Required attributes (typically set from IPS config)
    ---------------------------------------------------
    - ``OPTIONS_INP``: path to the BOUT.inp/options file in the work dir.
    - ``GRIDFILE``: path to the BOUT++ grid file in the work dir.
    - ``BIN_PATH``: path to the BOUT++/Hermes executable.
    - ``NPROC``: requested processor count (may be reduced to a valid split).

    Optional attributes
    -------------------
    - ``RESTART_TARFILE``: a tarball containing restart files. If present,
      it is extracted before launch, and an updated tarball is written after
      the run completes.

    Notes
    -----
    IPS services should not be called from ``__init__``. Any logging or other
    services interaction belongs in ``init(...)`` or later lifecycle methods.
    """

    def __init__(self, services, config):
        """Construct the component.

        IPS injects configuration values as attributes before calling
        ``__init__``. This constructor does not perform validation or logging
        because IPS services should not be used in ``__init__``.
        """
        super().__init__(services, config)

    def init(self, timestamp=0.0, **kwargs):
        """IPS lifecycle hook called before the first step.

        Subclasses may override this, but should typically call
        ``super().init(...)``.
        """
        self.services.info(f"Created {self.__class__}")

    def step(self, timestamp=0.0):
        """Run a BOUT++ simulation in the IPS working directory.

        The caller/subclass must set ``OPTIONS_INP``, ``GRIDFILE``, ``BIN_PATH``,
        and ``NPROC`` before calling this method.
        """
        self.services.info(f"BOUT++ step {timestamp}")

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
            self.services.info(f"Extracting restart tarfile '{self.RESTART_TARFILE}'")
            with tarfile.open(self.RESTART_TARFILE, "r:*") as tar:
                tar.extractall()

        self.services.info(f"Running executable  : {self.BIN_PATH}")

        cwd = self.services.get_working_dir()
        self.services.info(f"Working directory   : {cwd}")

        ncpu = self.num_processors(int(self.NPROC))
        self.services.info(f"Number of processors: {ncpu}")

        # Command line argument "-f <option file> -d <data file>"
        command_line_args = f"-f {self.OPTIONS_INP} -d {cwd}"
        if restarting:
            command_line_args += " restart"

        # Run simulation
        self.services.info(f"Arguments           : {command_line_args}")
        task_id = self.services.launch_task(
            ncpu, cwd, self.BIN_PATH, command_line_args, logfile="bout.log"
        )
        retcode = self.services.wait_task(task_id)

        # Tar restart file
        restart_files = list(Path.cwd().glob("BOUT.restart*.nc")) + list(
            Path.cwd().glob("BOUT.restart.bp")
        )
        self.services.info(f"Saving restart files {restart_files}")

        restart_tarfile = self.RESTART_TARFILE if restarting else "BOUT.restart.tar"
        with tarfile.open(restart_tarfile, "w") as tar:
            for file in restart_files:
                # arcname ensures the file doesn't store the full absolute path
                tar.add(file, arcname=file.name)
        self.services.info(f"Restarts saved to tarfile '{restart_tarfile}'")

        # Next step use this tarfile
        # The content of the working directory persist between `step`
        # calls so this restart file will be present next call.
        self.RESTART_TARFILE = restart_tarfile

        self.services.info("Finished BOUT++ step")

    def num_processors(self, max_nprocs: int) -> int:
        """Choose a valid processor count <= ``max_nprocs`` for ``self.GRIDFILE``.

        Reads mesh metadata from ``self.GRIDFILE`` and searches for the largest
        processor count not exceeding ``max_nprocs`` that satisfies BOUT++'s
        decomposition constraints.
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
