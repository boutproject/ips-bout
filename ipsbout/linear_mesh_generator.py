from ipsframework import Component
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)


def imas_coils(pf_active):
    """Extract coil positions and total ampere-turn currents from an IMAS ``pf_active`` IDS.

    Resolves the supply → circuit → coil connectivity graph to compute
    the effective current seen by each coil, defined as::

        coil_current[i] = supply_current[supply_index] * turns_with_sign[i]

    where ``supply_current`` is the per-filament output of the power supply
    and ``turns_with_sign`` is the signed filament count of the coil
    element (positive for co-current, negative for counter-current
    windings).

    Only the first element of each coil (``coil["element"][0]``) is read,
    and its geometry is assumed to be of type ``rectangle``, providing a
    centroid ``r`` and ``z``.  Multi-element coils and non-rectangular
    geometry types are not supported.

    Parameters
    ----------
    pf_active : dict
        IMAS ``pf_active`` IDS loaded from the machine JSON file.  Must
        contain the following keys:

        ``"supply"`` : list of dict
            Each entry must have ``supply["current"]["data"]`` (float,
            amperes), giving the per-filament supply current.

        ``"coil"`` : list of dict
            Each entry must have
            ``coil["element"][0]["geometry"]["rectangle"]["r"]`` (float, m),
            ``coil["element"][0]["geometry"]["rectangle"]["z"]`` (float, m),
            and ``coil["element"][0]["turns_with_sign"]`` (float).

        ``"circuit"`` : list of dict
            Each entry must have ``circuit["supply_index"]`` (int, 0-based
            index into ``"supply"``) and ``circuit["coil_indices"]`` (list
            of int, 0-based indices into ``"coil"``).

    Returns
    -------
    dict
        Dictionary with three 1-D numpy arrays, all of length
        ``len(pf_active["coil"])``:

        ``"r"`` : numpy.ndarray
            Radial positions of coil centroids in metres.

        ``"z"`` : numpy.ndarray
            Axial positions of coil centroids in metres.

        ``"current"`` : numpy.ndarray
            Total ampere-turn current of each coil in amperes
            (``turns_with_sign * supply_current``).
    """
    # Power supplies
    supply_currents = [supply["current"]["data"] for supply in pf_active["supply"]]

    # Coils
    coil_rs = []
    coil_zs = []
    coil_turns = []
    for coil in pf_active["coil"]:
        coil_rs.append(coil["element"][0]["geometry"]["rectangle"]["r"])
        coil_zs.append(coil["element"][0]["geometry"]["rectangle"]["z"])
        coil_turns.append(coil["element"][0]["turns_with_sign"])
    coil_rs = np.array(coil_rs)
    coil_zs = np.array(coil_zs)
    coil_turns = np.array(coil_turns)
    coil_currents = np.zeros(len(coil_turns))

    # Connections
    for circuit in pf_active["circuit"]:
        supply_current = supply_currents[circuit["supply_index"]]
        coil_currents[circuit["coil_indices"]] = supply_current
    coil_currents *= coil_turns

    return {"r": coil_rs, "z": coil_zs, "current": coil_currents}


def calc_penalty_mask(grid_data, wall_rz):
    """Compute a fractional wall-penalty mask for each mesh cell.

    Determines which cells in the computational mesh lie fully or partially
    outside the physical wall boundary, for use with Hermes-3 penalisation
    boundary conditions.  The wall is represented as a closed polygon in
    the (R, Z) plane that is revolved about the axis of symmetry.

    Cell exterior status is determined by a ray-casting test: a ray is
    drawn from a fixed interior reference point ``p0 = (R=0.01, Z=1.0)``
    to each cell-face vertex, and the number of wall-segment intersections
    is counted.  An odd count indicates the vertex lies outside the wall.

    The penalty value for each cell is:

    * ``0.0`` — cell is entirely inside the wall.
    * ``1.0`` — both axial face vertices (``p1`` and ``p2``) lie outside
        the wall.
    * ``(0, 1)`` — the cell straddles the wall; the value is the fraction
        of the cell edge (from the outside vertex to the wall intersection)
        relative to the total cell-edge length.

    The mask is computed on the axial (y-direction) cell faces, using the
    lower-face coordinates ``Rxy_ylow`` and ``Zxy_ylow`` for face vertices
    at ``j`` and ``j+1``.  Adapted from the Hypnotoad mesh generator.

    Parameters
    ----------
    grid_data : dict
        Dictionary of grid arrays as returned by
        ``AxisymMirrorMesh.orthogonal_mesh()``.  The following keys are
        accessed:

        ``"Rxy"`` : numpy.ndarray, shape (nx, ny)
            Radial coordinate of cell centres, used only to determine the
            grid shape.

        ``"Rxy_ylow"`` : numpy.ndarray, shape (nx, ny+1)
            Radial coordinate of the lower (y-direction) cell faces.

        ``"Zxy_ylow"`` : numpy.ndarray, shape (nx, ny+1)
            Axial coordinate of the lower (y-direction) cell faces.

    wall_rz : numpy.ndarray, shape (N, 2)
        Ordered vertices of the wall polygon in the (R, Z) plane, where
        column 0 is R (metres) and column 1 is Z (metres).  The polygon
        should be open (first and last vertices need not coincide) and is
        interpreted as a polyline by the Hypnotoad intersection routines.

    Returns
    -------
    numpy.ndarray, shape (nx, ny)
        Penalty mask array.  Values are in the range ``[0.0, 1.0]``.
        A value of ``0.0`` means the cell is entirely inside the wall; a
        value of ``1.0`` means it is entirely outside.  Intermediate
        values indicate partial wall intersection.
    """
    from hypnotoad import Point2D
    from hypnotoad.core.equilibrium import find_intersections, calc_distance

    Rxy = grid_data["Rxy"]
    Rxy_ylow = grid_data["Rxy_ylow"]
    Zxy_ylow = grid_data["Zxy_ylow"]

    penalty_mask = np.zeros(Rxy.shape)

    # A point inside the domain.
    p0 = Point2D(0.01, 1.0)

    for i in range(Rxy.shape[0]):
        for j in range(Rxy.shape[1]):
            # Check if cell Y edges are outside the wall
            p1 = Point2D(Rxy_ylow[i, j], Zxy_ylow[i, j])
            intersects = find_intersections(wall_rz, p0, p1)
            p1_outside = False if intersects is None else intersects.shape[0] % 2 == 1

            p2 = Point2D(Rxy_ylow[i, j + 1], Zxy_ylow[i, j + 1])
            intersects = find_intersections(wall_rz, p0, p2)
            p2_outside = False if intersects is None else intersects.shape[0] % 2 == 1

            if p1_outside and p2_outside:
                # Both ends of the cell are outside the wall
                penalty_mask[i, j] = 1.0
            elif p1_outside or p2_outside:
                # Cell crosses the wall
                intersects = find_intersections(wall_rz, p1, p2)
                if intersects is None:
                    # Something odd going on
                    continue
                pi = Point2D(intersects[0, 0], intersects[0, 1])  # Intersection point
                penalty_mask[i, j] = calc_distance(
                    p1 if p1_outside else p2, pi
                ) / calc_distance(p1, p2)
    return penalty_mask


class linear_mesh_generator(Component):
    """IPS worker component that generates a BOUT++ mesh for a linear plasma device.

    Reads a machine description (coil geometry and wall outline) from an
    IMAS-compatible JSON file, computes an orthogonal axisymmetric magnetic
    mirror mesh using :mod:`MirrorMesh`, appends a wall penalty mask, and
    writes the result to a BOUT++-compatible NetCDF grid file.

    The mesh is constructed from field-line-following coordinates aligned to
    the mirror magnetic field computed from the specified coil set.  A penalty
    mask is then added to indicate cells that lie outside the physical wall
    boundary, for use by Hermes-3 penalisation boundary conditions.

    This component should be registered as a worker under an IPS ``GRIDGEN``
    port.  A driver must call :meth:`step` to perform the mesh generation.
    The ``init`` and ``finalize`` lifecycle methods are inherited from
    :class:`ipsframework.Component` and perform no additional work.

    **IPS configuration parameters**

    All parameters are read from the component's section of the IPS
    simulation config file.  String values are injected by the framework as
    instance attributes before :meth:`__init__` is called.

    ``MACHINE_FILE`` : str
        Path to the IMAS-compatible JSON file describing the machine.  Must
        contain ``pf_active`` and ``wall`` top-level keys conforming to the
        IMAS ``pf_active`` and ``wall`` IDS structures respectively.  Listed
        in ``INPUT_FILES`` so that IPS stages it into the working directory.

    ``GRIDFILE`` : str
        Path of the BOUT++ NetCDF grid file to be created.  Listed in
        ``OUTPUT_FILES`` so that IPS stages it to the results directory after
        the step completes.

    ``N_RADIAL_CELLS`` : int
        Number of cells in the radial (cross-field) direction.  Must be a
        positive integer.

    ``N_AXIAL_CELLS`` : int
        Number of cells in the axial (along-field) direction.  Must be a
        positive integer.

    ``N_AZIMUTHAL_CELLS`` : INT
        Number of cells in the azimuthal (symmetry) direction. Must be a
        positive integer.

    ``R_MIN`` : float
        Minimum radius of the mesh domain in metres.  Must satisfy
        ``0 <= R_MIN < R_MAX``.

    ``R_MAX`` : float
        Maximum radius of the mesh domain in metres.

    ``Z_MIN`` : float
        Minimum axial coordinate of the mesh domain in metres.  Must satisfy
        ``Z_MIN < Z_MAX`` and both values must lie within the axial extent of
        the wall outline.

    ``Z_MAX`` : float
        Maximum axial coordinate of the mesh domain in metres.

    **Typical config file excerpt**::

        [GRIDGEN]
            CLASS = workers
            NAME = linear_mesh_generator
            NPROC = 1
            MODULE = ipsbout.linear_mesh_generator
            MACHINE_FILE = ProtoMPEX_G.json
            GRIDFILE = proto_mpex.nc
            INPUT_FILES = ${MACHINE_FILE}
            OUTPUT_FILES = ${GRIDFILE}
            N_RADIAL_CELLS = 64
            N_AXIAL_CELLS = 64
            N_AZIMUTHAL_CELLS = 64
            R_MIN = 0.05
            R_MAX = 0.3
            Z_MIN = 0.5
            Z_MAX = 4.2

    Raises
    ------
    ValueError
        If any required configuration parameter is absent or empty at
        construction time.
    """

    def __init__(self, services, config):
        """Initialise the component and validate required configuration parameters.

        Called by the IPS framework after injecting configuration entries from
        the simulation config file as instance attributes.  Validates that all
        required parameters are present and non-empty, and converts the numeric
        string values to their appropriate Python types, storing them as private
        attributes (``self.nradial``, ``self.naxial``, ``self.r_min``, etc.)
        for use in :meth:`step`.

        Parameters
        ----------
        services : ipsframework.ServicesProxy
            IPS services proxy providing access to framework facilities such as
            working-directory management, task launching, and file staging.
        config : configparser.SectionProxy
            The component's configuration section, as parsed from the IPS
            simulation config file.  The framework uses this to inject
            attributes onto ``self`` before this method is called.

        Raises
        ------
        ValueError
            If ``MACHINE_FILE``, ``GRIDFILE``, ``N_RADIAL_CELLS``,
            ``N_AXIAL_CELLS``, ``R_MIN``, ``R_MAX``, ``Z_MIN``, or ``Z_MAX``
            is absent from the configuration or is set to an empty string.
        """
        super().__init__(services, config)
        logger.info(f"Created {self.__class__}")

        if (not hasattr(self, "MACHINE_FILE")) or (self.MACHINE_FILE == ""):
            raise ValueError(
                "MACHINE_FILE must be set to the machine description JSON file"
            )

        if (not hasattr(self, "GRIDFILE")) or (self.GRIDFILE == ""):
            raise ValueError("GRIDFILE must be set to the output grid file.")

        if (not hasattr(self, "N_RADIAL_CELLS")) or (self.N_RADIAL_CELLS == ""):
            raise ValueError(
                "N_RADIAL_CELLS must be set to the number of radial cells."
            )
        self.nradial = int(self.N_RADIAL_CELLS)

        if (not hasattr(self, "N_AXIAL_CELLS")) or (self.N_AXIAL_CELLS == ""):
            raise ValueError("N_AXIAL_CELLS must be set to the number of axial cells.")
        self.naxial = int(self.N_AXIAL_CELLS)

        if (not hasattr(self, "N_AZIMUTHAL_CELLS")) or (self.N_AZIMUTHAL_CELLS == ""):
            raise ValueError("N_AZIMUTHAL_CELLS must be set to the number of azimuthal cells.")
        self.nazimuthal = int(self.N_AZIMUTHAL_CELLS)

        if (not hasattr(self, "R_MIN")) or (self.R_MIN == ""):
            raise ValueError("R_MIN must be set to the minimum radius in meters.")
        self.r_min = float(self.R_MIN)

        if (not hasattr(self, "R_MAX")) or (self.R_MAX == ""):
            raise ValueError("R_MAX must be set to the maximum radius in meters.")
        self.r_max = float(self.R_MAX)

        if (not hasattr(self, "Z_MIN")) or (self.Z_MIN == ""):
            raise ValueError(
                "Z_MIN must be set to the minimum axial location in meters."
            )
        self.z_min = float(self.Z_MIN)

        if (not hasattr(self, "Z_MAX")) or (self.Z_MAX == ""):
            raise ValueError(
                "Z_MAX must be set to the maximum axial location in meters."
            )
        self.z_max = float(self.Z_MAX)

    def step(self, timestamp=0.0):
        """Generate the BOUT++ mesh and write it to the grid file.

        This is the primary execution method of the component.  It performs
        the following sequence:

        1. Reads the machine description from ``MACHINE_FILE`` (staged into
           the working directory by IPS).
        2. Extracts coil geometry and currents from the ``pf_active`` IDS via
           :meth:`imas_coils`.
        3. Extracts the wall outline from the ``wall`` IDS.
        4. Constructs an orthogonal axisymmetric mirror mesh using
           :class:`MirrorMesh.AxisymMirrorMesh`, following field lines of the
           magnetic field computed from the coil set.
        5. Writes the mesh to the NetCDF file specified by ``GRIDFILE`` via
           :meth:`MirrorMesh.AxisymMirrorMesh.generate_hermes3_mesh`.
        6. Computes a fractional wall-penalty mask via :meth:`calcPenaltyMask`
           and appends it to the grid file as the variable ``penalty_mask``.

        Parameters
        ----------
        timestamp : float, optional
            IPS simulation timestamp passed by the driver.  Not used
            directly by this component, but forwarded by the framework.
            Defaults to ``0.0``.

        Raises
        ------
        FileNotFoundError
            If ``MACHINE_FILE`` does not exist in the working directory.
        json.JSONDecodeError
            If ``MACHINE_FILE`` is not valid JSON.
        KeyError
            If the expected ``pf_active`` or ``wall`` IDS keys are absent
            from the machine description JSON.
        RuntimeError
            If :class:`MirrorMesh.AxisymMirrorMesh` fails to converge on an
            orthogonal mesh for the given coil configuration and domain bounds.
        """
        from . import MirrorMesh as mm
        from boututils.datafile import DataFile

        logger.debug(f"INPUT_FILES: {self.INPUT_FILES}")
        logger.debug(f"OUTPUT_FILES: {self.OUTPUT_FILES}")

        with open(self.MACHINE_FILE, "r") as f:
            machine_data = json.load(f)

        # IMAS pf_active IDS
        coils = imas_coils(machine_data["pf_active"])

        # IMAS wall IDS
        wall = machine_data["wall"]["description_2d"][0]["limiter"]["unit"][0][
            "outline"
        ]
        wall_rz = np.stack((wall["r"], wall["z"]), axis=-1)  # 2D array

        AMM = mm.AxisymMirrorMesh(
            I_coil=coils["current"],
            z_coil=coils["z"],
            a_coil=coils["r"],
            nrho=self.nradial,
            nz=self.naxial,
            ntheta=self.nazimuthal,
            rho_range=[self.r_min, self.r_max],
            z_range=[self.z_min, self.z_max],
            radial_distance="physical",
        )
        grid_data = AMM.orthogonal_mesh()
        AMM.generate_hermes3_mesh(grid_data, save_dir=self.GRIDFILE)

        penalty_mask = calc_penalty_mask(grid_data, wall_rz)

        with DataFile(self.GRIDFILE, write=True) as f:
            f["penalty_mask"] = penalty_mask

        self.services.stage_output_files(timestamp, self.OUTPUT_FILES)
        self.services.update_state()
