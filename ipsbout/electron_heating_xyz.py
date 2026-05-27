from ipsframework import Component
from boututils.datafile import DataFile
import numpy as np

import logging

logger = logging.getLogger(__name__)


class electron_heating_xyz(Component):
    """IPS worker component that maps an electron heating profile onto a BOUT++ mesh.

    Reads a volumetric electron heating power density from a whitespace-delimited
    text file whose columns are ``(x, y, z, Q)`` in Cartesian coordinates
    (metres, W/m³), interpolates it onto the 3D cylindrical BOUT++ mesh, scales
    the result so that the volume-integrated power matches a prescribed total,
    and writes the resulting electron pressure source ``Pe_src`` back into the
    BOUT++ grid NetCDF file.

    The heating data are given in an arbitrary Cartesian frame that may be
    axially shifted relative to the machine coordinate system.  The parameter
    ``AXIAL_OFFSET`` corrects for this shift: the axial coordinate of each data
    point is taken as ``z_data + AXIAL_OFFSET`` when locating it on the mesh.

    The quantity written to the grid file is the electron pressure source
    ``Pe_src`` in pascals per second.  The corresponding electron heating power
    density is ``(3/2) * Pe_src`` W/m³.  Hermes-3 expects ``Pe_src`` in this
    form as a source term in the electron pressure equation.

    This component modifies ``GRIDFILE`` in-place; it must therefore be listed
    in both ``INPUT_FILES`` and ``OUTPUT_FILES`` so that IPS stages the file
    into the working directory before ``step`` runs and collects it afterwards.

    **IPS configuration parameters**

    ``GRIDFILE`` : str
        Path of the BOUT++ NetCDF grid file to be modified.  Must contain the
        fields ``nz``, ``Rxy``, ``Zxy``, ``J``, ``dx``, and ``dy``.  Listed in
        both ``INPUT_FILES`` and ``OUTPUT_FILES``.

    ``HEATING_FILE`` : str
        Path of the whitespace-delimited text file containing the heating
        profile.  Lines beginning with ``%`` are treated as comments.  Each
        data row must have at least four columns: ``x`` (m), ``y`` (m),
        ``z`` (m), and ``Q`` (W/m³), where ``(x, y, z)`` are Cartesian
        coordinates in the heating-profile frame.

    ``AXIAL_OFFSET`` : float
        Axial shift in metres to apply to the heating-profile z coordinates
        before locating them on the machine mesh.  Set to ``0.0`` if the
        heating data are already in machine coordinates.

    ``TOTAL_POWER`` : float
        Target volume-integrated electron heating power in watts.  The
        interpolated profile is rescaled so that its integral over the mesh
        equals this value exactly.

    **Typical config file excerpt**::

        [ELECTRON_HEATING]
            CLASS = workers
            NAME = electron_heating_xyz
            NPROC = 1
            MODULE = ipsbout.electron_heating_xyz
            GRIDFILE = proto_mpex.nc
            HEATING_FILE = helicon_profile.txt
            AXIAL_OFFSET = 1.745
            TOTAL_POWER = 16e3
            INPUT_FILES = ${HEATING_FILE}
            STATE_FILES = ${GRIDFILE}

    Raises
    ------
    ValueError
        If any required configuration parameter is absent or empty, or if the
        interpolated heating profile integrates to zero or a negative value
        over the mesh (indicating that the profile does not overlap the domain,
        e.g. due to an incorrect ``AXIAL_OFFSET``).
    """

    def __init__(self, services, config):
        """Initialise the component and validate required configuration parameters.

        Called by the IPS framework after injecting configuration entries as
        instance attributes.  Validates that all required parameters are present
        and non-empty, and converts numeric string values to floats, storing
        them as ``self.axial_offset`` and ``self.total_power``.

        Parameters
        ----------
        services : ipsframework.ServicesProxy
            IPS services proxy providing framework facilities.
        config : configparser.SectionProxy
            The component's configuration section from the IPS simulation
            config file.

        Raises
        ------
        ValueError
            If ``GRIDFILE``, ``HEATING_FILE``, ``AXIAL_OFFSET``, or
            ``TOTAL_POWER`` is absent from the configuration or is set to an
            empty string.
        """
        super().__init__(services, config)
        logger.info(f"Created {self.__class__}")

        if (not hasattr(self, "GRIDFILE")) or (self.GRIDFILE == ""):
            raise ValueError(
                "GRIDFILE must be set to the grid file that will be modified."
            )

        if (not hasattr(self, "HEATING_FILE")) or (self.HEATING_FILE == ""):
            raise ValueError(
                "HEATING_FILE must be set to the heating profile in XYZ text format"
            )

        if (not hasattr(self, "AXIAL_OFFSET")) or (self.AXIAL_OFFSET == ""):
            raise ValueError(
                "AXIAL_OFFSET must be set to the offset of the heating profile in the axial direction"
            )
        self.axial_offset = float(self.AXIAL_OFFSET)

        if (not hasattr(self, "TOTAL_POWER")) or (self.TOTAL_POWER == ""):
            raise ValueError(
                "TOTAL_POWER must be set to the total heating power on the mesh in Watts"
            )
        self.total_power = float(self.TOTAL_POWER)

    @staticmethod
    def _load_heating_data(heating_file, axial_offset):
        """Load the Cartesian heating profile and apply the axial coordinate offset.

        Reads a whitespace-delimited text file whose columns are
        ``(x, y, z, Q)``, where ``(x, y, z)`` are Cartesian coordinates in
        metres and ``Q`` is the heating power density in W/m³.  Lines
        beginning with ``%`` are treated as comments.

        The axial offset is applied to the z column so that the returned
        coordinates are in the machine frame::

            z_machine = z_file + axial_offset

        Parameters
        ----------
        heating_file : str
            Path to the heating profile text file.
        axial_offset : float
            Axial shift in metres to add to the file z coordinates.

        Returns
        -------
        xyz : numpy.ndarray, shape (N, 3)
            Cartesian coordinates of the heating data points in the machine
            frame, columns ``(x, y, z_machine)``.
        Q : numpy.ndarray, shape (N,)
            Heating power density at each data point in W/m³.
        """
        data = np.loadtxt(heating_file, comments="%")
        xyz = data[:, :3].copy()
        xyz[:, 2] += axial_offset
        Q = data[:, 3]
        return xyz, Q

    @staticmethod
    def _build_3d_mesh(Rxy, Zxy, J, dx, dy, nz):
        """Expand a 2-D axisymmetric mesh into 3-D Cartesian and volume arrays.

        Tiles the (R, Z) mesh over ``nz`` evenly-spaced toroidal angles
        spanning ``[0, 2π)``, converting each (R, θ, Z) point to Cartesian
        coordinates ``(x, y, z)``.  Also computes the cell volume for every
        3-D cell.

        Parameters
        ----------
        Rxy : numpy.ndarray, shape (nx, ny)
            Radial coordinate of each 2-D mesh cell centre in metres.
        Zxy : numpy.ndarray, shape (nx, ny)
            Axial coordinate of each 2-D mesh cell centre in metres.
        J : numpy.ndarray, shape (nx, ny)
            Jacobian of the mesh coordinate transformation.
        dx : numpy.ndarray, shape (nx, ny)
            Radial cell width in mesh coordinates.
        dy : numpy.ndarray, shape (nx, ny)
            Axial cell width in mesh coordinates.
        nz : int
            Number of toroidal grid points.

        Returns
        -------
        grid_x : numpy.ndarray, shape (nx, ny, nz)
            Cartesian x coordinate of each 3-D cell centre in metres.
        grid_y : numpy.ndarray, shape (nx, ny, nz)
            Cartesian y coordinate of each 3-D cell centre in metres.
        grid_z : numpy.ndarray, shape (nx, ny, nz)
            Axial coordinate of each 3-D cell centre in metres
            (identical for all toroidal indices).
        dV_xyz : numpy.ndarray, shape (nx, ny, nz)
            Volume of each 3-D cell in m³.
        """
        dz = 2 * np.pi / nz
        dV = J * dx * dy * dz
        dV_xyz = np.repeat(dV[..., np.newaxis], nz, axis=-1)

        thetas = np.linspace(0, 2 * np.pi, nz, endpoint=False)

        grid_r = np.repeat(Rxy[..., np.newaxis], nz, axis=-1)
        grid_z = np.repeat(Zxy[..., np.newaxis], nz, axis=-1)
        grid_theta = np.tile(thetas, Rxy.shape + (1,))

        grid_x = grid_r * np.cos(grid_theta)
        grid_y = grid_r * np.sin(grid_theta)

        return grid_x, grid_y, grid_z, dV_xyz

    @staticmethod
    def _interpolate_to_mesh(xyz, Q, grid_x, grid_y, grid_z, dV_xyz):
        """Interpolate scattered heating data onto the 3-D mesh and normalise.

        Identifies the axial index range of the mesh that overlaps with the
        heating data, performs linear scattered interpolation within that
        range, and returns the interpolated power density together with the
        bounding indices and the pre-normalisation integrated power.

        Cells outside the convex hull of the source data are assigned a power
        density of zero by the interpolator.

        Parameters
        ----------
        xyz : numpy.ndarray, shape (N, 3)
            Cartesian coordinates of the heating data points in the machine
            frame, columns ``(x, y, z)``.
        Q : numpy.ndarray, shape (N,)
            Heating power density at each data point in W/m³.
        grid_x : numpy.ndarray, shape (nx, ny, nz)
            Cartesian x coordinate of each mesh cell centre.
        grid_y : numpy.ndarray, shape (nx, ny, nz)
            Cartesian y coordinate of each mesh cell centre.
        grid_z : numpy.ndarray, shape (nx, ny, nz)
            Axial coordinate of each mesh cell centre.
        dV_xyz : numpy.ndarray, shape (nx, ny, nz)
            Volume of each mesh cell in m³.

        Returns
        -------
        grid_values : numpy.ndarray, shape (nx, ny_overlap, nz)
            Interpolated heating power density in W/m³ over the axial overlap
            region ``[ymin, ymax+1]``.
        ymin : int
            First axial mesh index within the heating data extent (inclusive).
        ymax : int
            Last axial mesh index within the heating data extent (inclusive).
        total_power : float
            Volume-integrated power of the unscaled interpolated profile in W.

        Raises
        ------
        ValueError
            If the volume-integrated power over the mesh is zero or negative,
            indicating that the heating profile does not overlap the mesh
            domain.
        """
        from scipy.interpolate import griddata

        # Use the axial coordinate at the first radial and toroidal index as a
        # representative 1-D axial axis.  For a structured mesh Zxy is
        # independent of the radial index so this is exact.
        z_axis = grid_z[0, :, 0]
        z_data = xyz[:, 2]

        # Find mesh indices nearest to the axial extent of the heating data.
        # ymax is inclusive: add 1 when slicing to include this cell.
        ymin = int(np.argmin(np.abs(z_axis - np.amin(z_data))))
        ymax = int(np.argmin(np.abs(z_axis - np.amax(z_data))))

        grid_values = griddata(
            xyz,
            Q,
            (
                grid_x[:, ymin : ymax + 1, :],
                grid_y[:, ymin : ymax + 1, :],
                grid_z[:, ymin : ymax + 1, :],
            ),
            method="linear",
            fill_value=0.0,
        )

        total_power = float(np.sum(grid_values * dV_xyz[:, ymin : ymax + 1, :]))

        if total_power <= 0.0:
            raise ValueError(
                f"Volume-integrated heating power is {total_power:.3e} W, which is zero "
                f"or negative.  The heating profile may not overlap the mesh domain.  "
                f"Check AXIAL_OFFSET (currently {xyz[0, 2] - z_data[0]:.4f} m) and that "
                f"the axial extent of the data [{np.amin(z_data):.3f}, {np.amax(z_data):.3f}] m "
                f"overlaps the mesh extent [{z_axis[0]:.3f}, {z_axis[-1]:.3f}] m."
            )

        return grid_values, ymin, ymax, total_power

    @staticmethod
    def _compute_Pe_src(grid_values, ymin, ymax, total_power, target_power, dV_xyz):
        """Scale the interpolated profile and convert to an electron pressure source.

        Rescales ``grid_values`` so that its volume integral equals
        ``target_power``, then converts from heating power density (W/m³) to
        the electron pressure source rate (Pa/s) expected by Hermes-3::

            Pe_src = (2/3) * Q_scaled

        where ``Q_scaled`` is the normalised power density.  The factor of 2/3
        follows from the relationship between the electron pressure equation
        source term and the energy source: the heating power density in W/m³
        equals ``(3/2) * Pe_src``.

        Parameters
        ----------
        grid_values : numpy.ndarray, shape (nx, ny_overlap, nz)
            Interpolated heating power density in W/m³ over the axial overlap
            region, as returned by :meth:`_interpolate_to_mesh`.
        ymin : int
            First axial mesh index of the overlap region (inclusive).
        ymax : int
            Last axial mesh index of the overlap region (inclusive).
        total_power : float
            Volume-integrated power of the unscaled profile in W, used for
            normalisation.
        target_power : float
            Desired volume-integrated electron heating power in W.
        dV_xyz : numpy.ndarray, shape (nx, ny, nz)
            Volume of each mesh cell in m³, used only to determine the full
            output array shape.

        Returns
        -------
        Pe_src : numpy.ndarray, shape (nx, ny, nz)
            Electron pressure source in Pa/s.  Non-zero only within the axial
            overlap region ``[ymin, ymax+1]``; zero elsewhere.
        """
        Q_scaled = grid_values * (target_power / total_power)

        # Pe_src is the electron pressure source in Pa/s.
        # The corresponding electron heating power density is (3/2) * Pe_src W/m³.
        Pe_src = np.zeros(dV_xyz.shape)
        Pe_src[:, ymin : ymax + 1, :] = (2.0 / 3.0) * Q_scaled

        return Pe_src

    def step(self, timestamp=0.0):
        """Interpolate the heating profile onto the mesh and write ``Pe_src``.

        Stages input files, delegates to :meth:`_load_heating_data`,
        :meth:`_build_3d_mesh`, :meth:`_interpolate_to_mesh`, and
        :meth:`_compute_Pe_src`, logs diagnostics, then writes the result to
        ``GRIDFILE`` and stages output files.

        Parameters
        ----------
        timestamp : float, optional
            IPS simulation timestamp passed by the driver.  Not used
            directly by this component.  Defaults to ``0.0``.

        Raises
        ------
        FileNotFoundError
            If ``HEATING_FILE`` or ``GRIDFILE`` does not exist in the working
            directory after IPS staging.
        ValueError
            If the volume-integrated power of the interpolated profile is zero
            or negative.  See :meth:`_interpolate_to_mesh` for details.
        """
        self.services.stage_input_files(self.INPUT_FILES)
        self.services.stage_state() # Fetch GRIDFILE to be modified

        xyz, Q = self._load_heating_data(self.HEATING_FILE, self.axial_offset)

        with DataFile(self.GRIDFILE) as grid:
            nz = grid["nz"]
            Rxy = grid["Rxy"]
            Zxy = grid["Zxy"]
            J = grid["J"]
            dx = grid["dx"]
            dy = grid["dy"]

        grid_x, grid_y, grid_z, dV_xyz = self._build_3d_mesh(Rxy, Zxy, J, dx, dy, nz)

        grid_values, ymin, ymax, total_power = self._interpolate_to_mesh(
            xyz, Q, grid_x, grid_y, grid_z, dV_xyz
        )

        logger.info(
            f"Maximum power density: input {np.amax(Q):.3e} W/m³ "
            f"-> interpolated {np.amax(grid_values):.3e} W/m³"
        )
        logger.info(f"Total domain volume: {np.sum(dV_xyz):.4e} m³")
        logger.info(f"Total input power before normalisation: {total_power:.4e} W")

        Pe_src = self._compute_Pe_src(
            grid_values, ymin, ymax, total_power, self.total_power, dV_xyz
        )

        with DataFile(self.GRIDFILE, write=True) as grid:
            grid["Pe_src"] = Pe_src

        logger.info(f"Pe_src written to {self.GRIDFILE}")

        self.services.stage_output_files(timestamp, self.OUTPUT_FILES)
        self.services.update_state() # Update GRIDFILE in the state
