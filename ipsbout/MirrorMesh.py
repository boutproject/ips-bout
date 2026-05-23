"""
This module construct the magnetic field geometry and mesh points for axisymmetric mirror.

Last updated by Y. Fu (07/2025)
"""

import numpy as np
import xarray as xr
from scipy.special import ellipe, ellipk


class MagneticGeometry:
    """
    This class calculate all quantities related to the magnetic fields in cylindrical geometry. The
    coils are assumed to be circular, parallel to each other, and aligned with the z-axis. The theta
    component of B is assumed to be zero.
    """

    def __init__(
        self,
        I_coil: np.ndarray,  # Coil currents (A)
        z_coil: np.ndarray,  # Axial positions (m)
        a_coil: np.ndarray,  # Coil radii (m)
    ):
        """
        Store all coil data for calculation
        """

        # check shape
        if (I_coil.shape != z_coil.shape) or (I_coil.shape != a_coil.shape):
            print("Shapes are different for coil parameters!")
            return None

        # check the number of coils
        if I_coil.shape[0] == 0:
            print("Numebr of coils is at least 1!")
            return None

        # initialize coil data
        self.coils = {}
        for i in range(I_coil.shape[0]):
            self.coils[i] = {"I": I_coil[i], "z": z_coil[i], "a": a_coil[i]}

    def loop_current_coeff(
        self, rho, zz, a, sinty, epsilon_coil=1e-4, epsilon_axis=1e-6
    ):
        """
        Calculate the coefficients, alpha, beta, k, and C

        Arguments:
        ---------------
        epsilon:
            If the distance between the point and the coil is smaller than epsilon, then
            regularize the distance to be epsilon.
        """

        mu0 = 4.0e-7 * np.pi

        # regularize the integral near the current
        ind = np.where(zz**2.0 + (rho - a) ** 2.0 < epsilon_coil**2.0)
        zz[ind] = epsilon_coil
        rho[ind] = a + epsilon_coil

        # regularize the integral on axis
        ind = np.where(rho**2.0 < epsilon_axis**2.0)
        rho[ind] = epsilon_axis

        alpha2 = a**2.0 + rho**2.0 + zz**2.0 - 2.0 * a * rho
        beta2 = a**2.0 + rho**2.0 + zz**2.0 + 2.0 * a * rho
        beta = np.sqrt(beta2)
        k2 = 1.0 - alpha2 / beta2
        C = mu0 * sinty / np.pi

        return alpha2, beta2, beta, k2, C

    def magnetic_field_rho(self, rho: np.ndarray, z: np.ndarray):
        """
        Calculate the radial component of magnetic field
        """

        # check shape
        if rho.shape != z.shape:
            print("Shapes are different for rho and z!")
            return None

        # initialize the magnetic fields
        B_rho = np.zeros(rho.shape)

        # loop over all coils
        for i in range(len(self.coils.keys())):
            a = self.coils[i]["a"]
            sinty = self.coils[i]["I"]
            zz = z - self.coils[i]["z"]

            alpha2, beta2, beta, k2, C = self.loop_current_coeff(
                rho=rho, zz=zz, a=a, sinty=sinty
            )

            B_rho += (
                C
                * zz
                / (2.0 * alpha2 * beta * rho)
                * ((a**2.0 + rho**2.0 + zz**2.0) * ellipe(k2) - alpha2 * ellipk(k2))
            )

        return B_rho

    def magnetic_field_z(self, rho: np.ndarray, z: np.ndarray):
        """
        Calculate the z-component of magnetic field
        """

        # check shape
        if rho.shape != z.shape:
            print("Shapes are different for rho and z!")
            return None

        # initialize the magnetic fields
        B_z = np.zeros(rho.shape)

        # loop over all coils
        for i in range(len(self.coils.keys())):
            a = self.coils[i]["a"]
            sinty = self.coils[i]["I"]
            zz = z - self.coils[i]["z"]

            alpha2, beta2, beta, k2, C = self.loop_current_coeff(
                rho=rho, zz=zz, a=a, sinty=sinty
            )

            B_z += (
                C
                / (2.0 * alpha2 * beta)
                * ((a**2.0 - rho**2.0 - zz**2.0) * ellipe(k2) + alpha2 * ellipk(k2))
            )

        return B_z

    def magnetic_field(self, rho: np.ndarray, z: np.ndarray):
        """
        Calculate the magnetic field strength at given locations.
        """

        B_rho = self.magnetic_field_rho(rho, z)
        B_z = self.magnetic_field_z(rho, z)

        return B_rho, B_z

    def magnetic_field_strength(self, rho: np.ndarray, z: np.ndarray):
        """
        Calculate the magnitude of the magnetic field.
        """

        B_rho, B_z = self.magnetic_field(rho, z)

        return (B_rho**2.0 + B_z**2.0) ** 0.5

    def magnetic_field_direction(self, rho: np.ndarray, z: np.ndarray):
        """
        Calculate the unit vector b = mathbf{B} / B
        """

        B_rho, B_z = self.magnetic_field(rho, z)
        B = self.magnetic_field_strength(rho, z)

        return B_rho / B, B_z / B

    def magnetic_flux(self, rho: np.ndarray, z: np.ndarray):
        """
        Calculate the magnetic flux using scipy.integrate.quad_vec, which is slow but
        as accurate as analytical.
        """
        from scipy.integrate import quad_vec

        # Define the function to be integrated. Integration is carried over rr, so
        # rr has to be float and the first variable.
        def f(rr: float, zz: np.ndarray):
            _rr = np.zeros(zz.shape) + rr
            B_z = self.magnetic_field_z(_rr, zz)
            return rr * B_z

        # initialize all arraies
        rho_flat = np.ndarray.flatten(rho)
        rho_unique = np.unique(rho_flat)
        z_flat = np.ndarray.flatten(z)
        psi = np.zeros(rho_flat.shape)

        # calculate data in all points
        for i in range(len(rho_unique)):
            # only integrate once using quad_vec for each rho
            ind = np.where(rho_flat == rho_unique[i])
            psi[ind], err = quad_vec(f=f, a=0, b=rho_unique[i], args=(z_flat[ind],))

        return psi.reshape(rho.shape)

    def magnetic_curvature(self, rho: np.ndarray, z: np.ndarray, d=1.0e-6):
        """
        Calculate the magnetic curvature kappa = (b.grad) b
        """

        # calculate derivative along rho
        b_rho_p, b_z_p = self.magnetic_field_direction(rho + d, z)
        b_rho_m, b_z_m = self.magnetic_field_direction(rho - d, z)

        d_b_rho_d_rho = (b_rho_p - b_rho_m) / (2.0 * d)
        d_b_z_d_rho = (b_z_p - b_z_m) / (2.0 * d)

        # calculate derivative along z
        b_rho_p, b_z_p = self.magnetic_field_direction(rho, z + d)
        b_rho_m, b_z_m = self.magnetic_field_direction(rho, z - d)

        d_b_rho_d_z = (b_rho_p - b_rho_m) / (2.0 * d)
        d_b_z_d_z = (b_z_p - b_z_m) / (2.0 * d)

        # calculate kappa
        b_rho, b_z = self.magnetic_field_direction(rho, z)
        kappa_rho = b_rho * d_b_rho_d_rho + b_z * d_b_rho_d_z
        kappa_z = b_rho * d_b_z_d_rho + b_z * d_b_z_d_z

        return kappa_rho, kappa_z


class AxisymMirrorMesh(MagneticGeometry):
    """
    This class defines a general procedure of mesh generation for axisymmetric mirrors. BOUT++/Hermes-3
    convention for axisymmetric mirror: x - radial, y - field-line, z - azimuthal.

    The mesh data includes 2-D profiles of psi, R, Z on grid points and corners.
    TODO: add other self.radial_distances options
    """

    def __init__(
        self,
        I_coil,
        z_coil,
        a_coil,
        nrho: int = 64,
        nz: int = 256,
        ntheta: int = 1,
        rho_range: list = [0.01, 0.06],
        z_range: list = [-2.56, 2.56],
        radial_distance: str = "physical",
    ):
        """
        Initialize the parameters needed for mesh generation

        Arguments:
        ---------------
        I_coil, z_coil, a_coil:
            Coil settings from MagneticGeometry class
        nrho:
            Number of raidal cells without four ghost cells. Notice: nrho = nx - 4.
        nz:
            Number of field-line cells without four ghost cells. Notice: nz = ny.
        ntheta:
            Number of azimuthal cells without four ghost cells.
        rho_range:
            Simulation range in radial direction at the midplane (m), rho_range[0] < rho_range[1]
        z_range:
            Simulation range in axial diretion (m), z_range[0] < z_range[1]
        radial_distance:
            Method to define the distance between radial coordiante:
                'physical': Equal distance based on radius
                'flux': Equal distance based on magnetic flux
        """

        super().__init__(I_coil, z_coil, a_coil)
        self.nrho = nrho
        self.rho_range = rho_range
        self.nz = nz
        self.ntheta = ntheta
        self.z_range = z_range
        self.radial_distance = radial_distance

    def field_line_tracer_parallel(
        self, rho0: np.ndarray, z0: float = 0.0, z1: float = 1.0, dense_output=False
    ):
        """
        Calculate the steam line along the magnetic field direction, using scipy.integrate.solve_ivp.
        The default integration method is RK45.

        Arguments:
        ---------------
        rho0:
            The start rho value for each field line, which is allowed to be an array with different values.
        z0, z1:
            The start and end z value for each field line, which are assumed to be the same for all field lines.
        dense_output:
            solve_ivp setting, whether to return a continuous result. Only used for test

        Return:
        ---------------
        sol:
            Solution given by solve_ivp. sol.y[:,-1] is the final rho we want.
        """
        from scipy.integrate import solve_ivp

        # Define the function to be integrated. Integration is carried along z,
        # so z has to be the first variable and is float.
        def f(z: float, rho: np.ndarray):
            _z = np.zeros(rho.shape) + z
            B_rho, B_z = self.magnetic_field(rho=rho, z=_z)
            return B_rho / B_z

        sol = solve_ivp(
            fun=f,
            t_span=[z0, z1],
            y0=rho0,
            dense_output=dense_output,
            atol=1e-7,
            rtol=1e-7,
        )

        return sol

    def field_line_tracer_normal(self, z0: np.ndarray, rho0=0.0, psi1: float = 0.01):
        """
        Calculate the steam line along the normal direction of the magnetic field, using scipy.integrate.solve_ivp.
        The integration is carried from psi0 to psi1, which is the magnetic flux at the start and end points. By default,
        the start point is the magnetic axis with rho0 = psi0 = 0.
        The default integration method is RK45.

        Arguments:
        ---------------
        z0, rho0:
            The start z and rho value for each field line, which are allowed to be an array with different values.
        rho_max:
            The maximum rho value for field line integration, which is a dummy variable only used for integration.
        psi1:
            The maximum magnetic flux value for field line integration.

        Return:
        ---------------
        sol:
            Solution given by solve_ivp. np.sqrt(sol.y[:len(z0),-1]) is the final rho, and np.sqrt(sol.y[len(z0):,-1])
            is the final z.
        """
        from scipy.integrate import solve_ivp

        # Define the function to be integrated. The variable being integrated is psi.
        def f(psi: float, rho2_z: np.ndarray, epsilon=1e-6):
            # unpack the rho2_z
            N_rho = int(len(rho2_z) / 2.0)
            rho = np.sqrt(rho2_z[:N_rho])
            z = rho2_z[N_rho:]

            # regularize the integral on axis
            ind = np.where(rho**2.0 < epsilon**2.0)
            rho[ind] = epsilon

            # Calculate RHS
            B_rho, B_z = self.magnetic_field(rho=rho, z=z)
            B_norm = np.sqrt(B_rho**2 + B_z**2)

            return np.concatenate(
                [2.0 * B_z / B_norm**2.0, -B_rho / (B_norm**2.0 * rho)]
            )

        # integrate the function
        rho2_z0 = np.concatenate([np.zeros(len(z0)) + rho0**2.0, z0])
        sol = solve_ivp(
            fun=f,
            t_span=[0.0, psi1],
            y0=rho2_z0,
            dense_output=True,
            atol=1e-7,
            rtol=1e-7,
        )

        return sol

    def scaling_coefficient(
        self, z0: np.ndarray, rho0=0.0, psi1: float = 0.01, vacuum_field=True
    ):
        """
        Calculate the scaling coefficient h(r, z), which is given by an integral normal to the magnetic field.
        TODO: Calculate h for non-vacuum field.

        Arguments:
        ---------------
        z0:
            The start z value for each field line, which is allowed to be an array with different values.
        rho0:
            The start rho value for each field line, which is assumed to be the same and is only a float.
        psi1:
            The end psi value for each field line, which is assumed to be the same.
        vacuum_field:
            Whether the magnetic field is vacuum field. If True, then the scaling coefficient is 1.
        """

        if vacuum_field:
            return np.zeros(z0.shape) + 1.0
        else:
            raise NotImplementedError("Non-vacuum magnetic field is not implemented.")

    def field_line_tracer_parallel_scaled(
        self,
        rho0: float,
        z0: float = 0.0,
        z1: float = 1.0,
        dense_output=False,
        vacuum_field=True,
    ):
        """
        Calculate the scaled arc length along the magnetic field direction, using scipy.integrate.solve_ivp.
        The default integration method is RK45. Since the end z1 for each field line is not the same, we need to
        calculate the scaled arc length for each field line separately.
        TODO: Rewrite this function for non-vacuum magnetic field

        Arguments:
        ---------------
        rho0:
            The start rho value for the field line.
        z0, z1:
            The start and end z value for the field line.
        dense_output:
            solve_ivp setting, whether to return a continuous result. Only used for test

        Return:
        ---------------
        sol:
            Solution given by solve_ivp. sol.y[:,-1] is the final [rho, s] for each field line.
        """
        from scipy.integrate import solve_ivp

        # Define the function to be integrated. Integration is carried along z, so z has to be the first
        # variable and is float. The second variable is rho_s = (rho, s), whose shape is (2,).
        def f(z: float, rho_s: np.ndarray):
            # unpack the rho_s
            [rho, s] = rho_s

            # Calculate the magnetic field
            [B_rho], [B_z] = self.magnetic_field(rho=np.array([rho]), z=np.array([z]))
            B = np.sqrt(B_rho**2.0 + B_z**2.0)

            # Calculate the scaling coefficient h
            # TODO: need to be modified for non-vacuum field
            [h] = self.scaling_coefficient(
                z0=np.array([z]),
                rho0=np.array([rho]),
                psi1=1.0,
                vacuum_field=vacuum_field,
            )

            return np.array([B_rho / B_z, B * h * np.sqrt(1.0 + (B_rho / B_z) ** 2.0)])

        # integrate the function
        rho0_s0 = np.array([rho0, 0.0])
        sol = solve_ivp(
            fun=f,
            t_span=[z0, z1],
            y0=rho0_s0,
            dense_output=dense_output,
            atol=1e-7,
            rtol=1e-7,
        )

        return sol

    def interpolate_grids(self, grid_data: dict):
        """
        Interpolate the corner values to center values.

        Argument:
        -----
        grid_data: a dictionary contains corner values. Must include the following keys:
        Corner values: rho_xy_corner, Z_xy_corner.
        Other values: dx_ylow, dy_xlow.
        """
        from scipy.interpolate import griddata

        # --- Test necessary keys ---
        key_required = ["Rxy_corner", "Zxy_corner", "dx", "dy"]
        for key in key_required:
            if key not in grid_data:
                raise KeyError("Key {} is not in grid_data".format(key))

        grid_data_interpolated = {}

        # --- Interpolate the coordinate rho, Z to center, x_low, and y_low values ---
        rho_xy_corner = grid_data["Rxy_corner"]
        Z_xy_corner = grid_data["Zxy_corner"]

        rho_xy = (
            rho_xy_corner[1:, 1:]
            + rho_xy_corner[1:, :-1]
            + rho_xy_corner[:-1, 1:]
            + rho_xy_corner[:-1, :-1]
        ) / 4.0
        Z_xy = (
            Z_xy_corner[1:, 1:]
            + Z_xy_corner[1:, :-1]
            + Z_xy_corner[:-1, 1:]
            + Z_xy_corner[:-1, :-1]
        ) / 4.0

        rho_xy_xlow = (rho_xy_corner[:, 1:] + rho_xy_corner[:, :-1]) / 2.0
        Z_xy_xlow = (Z_xy_corner[:, 1:] + Z_xy_corner[:, :-1]) / 2.0

        rho_xy_ylow = (rho_xy_corner[1:, :] + rho_xy_corner[:-1, :]) / 2.0
        Z_xy_ylow = (Z_xy_corner[1:, :] + Z_xy_corner[:-1, :]) / 2.0

        # Store the interpolated values
        grid_data_interpolated.update(
            {
                "Rxy": rho_xy,
                "Zxy": Z_xy,
                "Rxy_xlow": rho_xy_xlow,
                "Zxy_xlow": Z_xy_xlow,
                "Rxy_ylow": rho_xy_ylow,
                "Zxy_ylow": Z_xy_ylow,
            }
        )

        # # --- Special case for dx_ylow ---
        # dx_ylow = grid_data['dx_ylow']
        # dx = griddata(points=(rho_xy_ylow.flatten(), Z_xy_ylow.flatten()),
        #               values=dx_ylow.flatten(),
        #               xi=(rho_xy.flatten(), Z_xy.flatten()),
        #               method='cubic').reshape(self.nrho+4, self.nz)

        # # Check whether dx is constant in y-direction (along the field line)
        # dx_error_along_y = ((dx.max(axis=1) - dx.min(axis=1))/dx.min(axis=1)).max()
        # print("dx error along y-direction (field line):", dx_error_along_y)

        # # Store the interpolated values
        # grid_data_interpolated.update({
        #     'dx': dx,
        # })

        # # --- Special case for dy_xlow ---
        # dy_xlow = grid_data['dy_xlow']
        # dy = griddata(points=(rho_xy_xlow.flatten(), Z_xy_xlow.flatten()),
        #               values=dy_xlow.flatten(),
        #               xi=(rho_xy.flatten(), Z_xy.flatten()),
        #               method='cubic').reshape(self.nrho+4, self.nz)

        # # Check whether dy is constant in x-direction (radial direction)
        # dy_error_along_x = ((dy.max(axis=0) - dy.min(axis=0))/dy.min(axis=0)).max()
        # print("dy error along x-direction (radial direction):", dy_error_along_x)

        # # Store the interpolated values
        # grid_data_interpolated.update({
        #     'dy': dy,
        # })

        # --- Check whether dx and dy are 1-D ---
        # Check whether dx is constant in y-direction (along the field line)
        dx = grid_data["dx"]
        if not np.all(dx.max(axis=1) == dx.min(axis=1)):
            print(
                "Warning: dx is not constant in y-direction. Max error = {:.2e}".format(
                    (dx.max(axis=1) - dx.min(axis=1)).max()
                )
            )

        # Check whether dy is constant in x-direction (radial direction)
        dy = grid_data["dy"]
        if not np.all(dy.max(axis=0) == dy.min(axis=0)):
            print(
                "Warning: dy is not constant in x-direction. Max error = {:.2e}".format(
                    (dy.max(axis=0) - dy.min(axis=0)).max()
                )
            )

        # --- All other corner values ---
        for key in grid_data.keys():
            if key not in key_required:
                # Test whether it is a corner value
                if "_corner" not in key:
                    raise ValueError("Key {} is not a corner value.".format(key))
                else:
                    params_name = key.replace("_corner", "")

                # Interpolation
                corner_data = grid_data[key]
                center_data = griddata(
                    points=(rho_xy_corner.flatten(), Z_xy_corner.flatten()),
                    values=corner_data.flatten(),
                    xi=(rho_xy.flatten(), Z_xy.flatten()),
                    method="cubic",
                ).reshape(self.nrho + 4, self.nz)
                x_low_data = griddata(
                    points=(rho_xy_corner.flatten(), Z_xy_corner.flatten()),
                    values=corner_data.flatten(),
                    xi=(rho_xy_xlow.flatten(), Z_xy_xlow.flatten()),
                    method="cubic",
                ).reshape(self.nrho + 5, self.nz)
                y_low_data = griddata(
                    points=(rho_xy_corner.flatten(), Z_xy_corner.flatten()),
                    values=corner_data.flatten(),
                    xi=(rho_xy_ylow.flatten(), Z_xy_ylow.flatten()),
                    method="cubic",
                ).reshape(self.nrho + 4, self.nz + 1)

                # Store the interpolated values
                grid_data_interpolated.update(
                    {
                        params_name: center_data,
                        params_name + "_xlow": x_low_data,
                        params_name + "_ylow": y_low_data,
                    }
                )

        # combine two dictionaries
        grid_data_interpolated.update(grid_data)

        return grid_data_interpolated

    def cylindrical_mesh(self, approximate_metric=True):
        """
        Generate mesh grid in (Psi, theta, Z) coordiante for grid corner points, where angle theta is omitted.
        For BOUT++/Hermes-3 convention, we have: x == radial == Psi, y == field-line == Z.
        Notice: Capitcal Z is used in cylindrical coordinate, while lower case z is used as a field-line coordinate.
        TODO: add other self.radial_distances options
        """

        # Define the shape of the corner values
        shape_corner = (self.nrho + 5, self.nz + 1)

        # Initialize the dictionray for grid data
        grid_data = {}

        # --- Calculate the rho, Z, and psi values for the mesh grid ---
        rho_xy_corner, Z_xy_corner, psi_xy_corner = [
            np.zeros(shape_corner) for _ in range(3)
        ]

        # Set the Z coordiante
        Z_xy_corner += np.linspace(
            self.z_range[0], self.z_range[1], self.nz + 1
        ).reshape((1, -1))

        # define the starting point of each field lines in the midplane
        if self.radial_distance == "physical":
            drho = (self.rho_range[1] - self.rho_range[0]) / self.nrho
            mid_plane_rho0_corner = np.linspace(
                self.rho_range[0] - 2.0 * drho,
                self.rho_range[1] + 2.0 * drho,
                self.nrho + 5,
            )

        # integrate the field line for each given z, for corner girds
        for j in range(self.nz + 1):
            _z = Z_xy_corner[0, j]
            sol = self.field_line_tracer_parallel(
                rho0=mid_plane_rho0_corner, z0=0.0, z1=_z
            )
            rho_xy_corner[:, j] = sol.y[:, -1]

        # calculate the flux function, which is only a function of the first variable
        for i in range(self.nrho + 5):
            ind_z = np.argmin(np.abs(Z_xy_corner[i, :]))
            psi = self.magnetic_flux(
                rho=np.array([rho_xy_corner[i, ind_z]]),
                z=np.array([Z_xy_corner[i, ind_z]]),
            )
            psi_xy_corner[i, :] = psi[0]

        # Store the corner values
        grid_data.update(
            {
                "Rxy_corner": rho_xy_corner,
                "Zxy_corner": Z_xy_corner,
                "psi_xy_corner": psi_xy_corner,
            }
        )

        # --- Calculate the differencing values ---
        dx_ylow = psi_xy_corner[1:, :] - psi_xy_corner[:-1, :]
        dy_xlow = Z_xy_corner[:, 1:] - Z_xy_corner[:, :-1]

        # Store the differencing values
        # Notte: strictly speaking, dx and dx_ylow are not the same, but they are very close. Same for dy and dy_low
        grid_data.update({"dx": dx_ylow[:, :-1], "dy": dy_xlow[:-1, :]})

        # --- Calculate the magnetic field quantities ---
        # Calculate the magnetic field quantities
        B_xy_corner = self.magnetic_field_strength(rho=rho_xy_corner, z=Z_xy_corner)
        Br_xy_corner, Bz_xy_corner = self.magnetic_field(
            rho=rho_xy_corner, z=Z_xy_corner
        )

        # Store the magnetic field quantities
        grid_data.update(
            {
                "B_xy_corner": B_xy_corner,
                "Br_xy_corner": Br_xy_corner,
                "Bz_xy_corner": Bz_xy_corner,
            }
        )

        # --- Calculate the metric tensor and Jacobian ---
        g12_corner, g13_corner, g23_corner = [np.zeros(shape_corner) for _ in range(3)]
        g_12_corner, g_13_corner, g_23_corner = [
            np.zeros(shape_corner) for _ in range(3)
        ]

        # Decide whether to use approximated metric (no off-diagonal terms) or exact metric (with off-diagonal terms)
        if approximate_metric:
            g11_corner = rho_xy_corner**2.0 * B_xy_corner**2.0
            g22_corner = np.zeros(shape_corner) + 1.0
            g33_corner = 1.0 / rho_xy_corner**2.0

            g_11_corner = 1.0 / (rho_xy_corner**2.0 * B_xy_corner**2.0)
            g_22_corner = np.zeros(shape_corner) + 1.0
            g_33_corner = rho_xy_corner**2.0

            J_corner = 1.0 / Bz_xy_corner

        else:
            g11_corner = rho_xy_corner**2.0 * B_xy_corner**2.0
            g22_corner = np.zeros(shape_corner) + 1.0
            g33_corner = 1.0 / rho_xy_corner**2.0
            g12_corner = -rho_xy_corner * Br_xy_corner

            g_11_corner = 1.0 / (rho_xy_corner**2.0 * Bz_xy_corner**2.0)
            g_22_corner = B_xy_corner**2.0 / Bz_xy_corner**2.0
            g_33_corner = rho_xy_corner**2.0
            g_12_corner = Br_xy_corner / (rho_xy_corner * Bz_xy_corner**2.0)

            J_corner = 1.0 / B_xy_corner

        # Store the metric tensor
        grid_data.update(
            {
                "g11_corner": g11_corner,
                "g22_corner": g22_corner,
                "g33_corner": g33_corner,
                "g12_corner": g12_corner,
                "g13_corner": g13_corner,
                "g23_corner": g23_corner,
                "g_11_corner": g_11_corner,
                "g_22_corner": g_22_corner,
                "g_33_corner": g_33_corner,
                "g_12_corner": g_12_corner,
                "g_13_corner": g_13_corner,
                "g_23_corner": g_23_corner,
                "J_corner": J_corner,
            }
        )

        # --- Calculate the curvatures ---
        bxcvx_corner, bxcvy_corner = [np.zeros(shape_corner) for _ in range(2)]

        # bxcvz = kappa / rho
        kappa_rho, kappa_z = self.magnetic_curvature(rho=rho_xy_corner, z=Z_xy_corner)
        b_rho, b_z = self.magnetic_field_direction(rho=rho_xy_corner, z=Z_xy_corner)
        bxcvz_corner = (kappa_rho * b_z - kappa_z * b_rho) / rho_xy_corner

        # Store the curvatures
        grid_data.update(
            {
                "bxcvx_corner": bxcvx_corner,
                "bxcvy_corner": bxcvy_corner,
                "bxcvz_corner": bxcvz_corner,
            }
        )

        # Interpolate the corner grids to the center, xlow, and ylow values
        grid_data_interpolated = self.interpolate_grids(grid_data)

        return grid_data_interpolated

    def orthogonal_mesh(
        self, scaled_arc_length_method="normal_constant", vacuum_field=True
    ):
        """
        Generate mesh grid in (Psi, theta, s) coordinate, where s is the scaled arc length along the magnetic field line.
        TODO: need to be modified for non-vacuum magnetic field.
        TODO: add other self.radial_distances options

        Arguments:
        ---------------
        scaled_arc_length_method:
            Method to calculate the scaled arc length:
                'parallel_integral': Calculate the scaled arc length by integrating the field line along the parallel direction
                'normal_constant': Only calculate the scaled arc length along axis, and assume normal direction are same.
        vacuum_field:
            Whether the magnetic field is vacuum field. If True, then the scaling coefficient is 1.
            If False, then the scaling coefficient is calculated by self.scaling_coefficient.
        """

        # Define the shape of the corner values
        shape_corner = (self.nrho + 5, self.nz + 1)

        # Initialize the dictionary for grid data
        grid_data = {}

        # --- Calculate the rho, Z, psi, and s values for the mesh grid ---
        rho_xy_corner, Z_xy_corner, psi_xy_corner, s_xy_corner = [
            np.zeros(shape_corner) for _ in range(4)
        ]

        # fine the psi at the midplane
        if self.radial_distance == "physical":
            drho = (self.rho_range[1] - self.rho_range[0]) / self.nrho
            mid_plane_rho0_corner = np.linspace(
                self.rho_range[0] - 2.0 * drho,
                self.rho_range[1] + 2.0 * drho,
                self.nrho + 5,
            )
            mid_plane_psi_corner = self.magnetic_flux(
                rho=mid_plane_rho0_corner, z=np.zeros(self.nrho + 5)
            )

        # define the starting rho on axis
        axis_z0_corner = np.linspace(self.z_range[0], self.z_range[1], self.nz + 1)

        # integrate along the normal direction of the field line
        for i in range(self.nrho + 5):
            sol = self.field_line_tracer_normal(
                z0=axis_z0_corner, psi1=mid_plane_psi_corner[i]
            )

            # unpack rho2 and z
            rho_xy_corner[i, :] = np.sqrt(sol.y[: self.nz + 1, -1])
            Z_xy_corner[i, :] = sol.y[self.nz + 1 :, -1]
            psi_xy_corner[i, :] = mid_plane_psi_corner[i]

        # calculate the scaled arc length
        ind_z_mid = np.argmin(np.abs(Z_xy_corner[0, :]))
        if scaled_arc_length_method == "parallel_integral":
            for i in range(self.nrho + 5):
                for j in range(self.nz + 1):
                    sol = self.field_line_tracer_parallel_scaled(
                        rho0=rho_xy_corner[i, ind_z_mid],
                        z0=0.0,
                        z1=Z_xy_corner[i, j],
                        vacuum_field=vacuum_field,
                    )
                    s_xy_corner[i, j] = sol.y[1, -1]
                print("i = {} / {}".format(i, self.nrho + 5), end="\r")
        elif scaled_arc_length_method == "normal_constant":
            for j in range(self.nz + 1):
                sol = self.field_line_tracer_parallel_scaled(
                    rho0=rho_xy_corner[0, ind_z_mid],
                    z0=0.0,
                    z1=Z_xy_corner[0, j],
                    vacuum_field=vacuum_field,
                )
                s_xy_corner[:, j] = sol.y[1, -1]
        else:
            raise ValueError(
                "Invalid scaled arc length method: {}".format(scaled_arc_length_method)
            )

        # Store the rho, z, psi, and s values
        grid_data.update(
            {
                "Rxy_corner": rho_xy_corner,
                "Zxy_corner": Z_xy_corner,
                "psi_xy_corner": psi_xy_corner,
                "s_xy_corner": s_xy_corner,
            }
        )

        # --- Calculate the differencing values ---
        dx_ylow = psi_xy_corner[1:, :] - psi_xy_corner[:-1, :]
        dy_xlow = s_xy_corner[:, 1:] - s_xy_corner[:, :-1]

        # Store the differencing values
        # Note: strictly speaking, dx and dx_ylow are not the same, but they are very close. Same for dy and dy_low
        grid_data.update({"dx": dx_ylow[:, :-1], "dy": dy_xlow[:-1, :]})

        # --- Calculate the magnetic field quantities ---
        # Calculate the magnetic field quantities
        B_xy_corner = self.magnetic_field_strength(rho=rho_xy_corner, z=Z_xy_corner)
        Br_xy_corner, Bz_xy_corner = self.magnetic_field(
            rho=rho_xy_corner, z=Z_xy_corner
        )

        # Store the magnetic field quantities
        grid_data.update(
            {
                "Bxy_corner": B_xy_corner,
                "Brxy_corner": Br_xy_corner,
                "Bzxy_corner": Bz_xy_corner,
            }
        )

        # --- Calculate the metric tensor and Jacobian ---
        if vacuum_field:
            h = 1.0
        else:
            raise NotImplementedError("Non-vacuum magnetic field is not implemented.")

        g11_corner = rho_xy_corner**2.0 * B_xy_corner**2.0
        g22_corner = h**2.0 * B_xy_corner**2.0
        g33_corner = 1.0 / rho_xy_corner**2.0

        g_11_corner = 1.0 / (rho_xy_corner**2.0 * B_xy_corner**2.0)
        g_22_corner = 1.0 / (h**2.0 * B_xy_corner**2.0)
        g_33_corner = rho_xy_corner**2.0

        J_corner = 1.0 / (h * B_xy_corner**2.0)

        # Store the metric tensor
        grid_data.update(
            {
                "g11_corner": g11_corner,
                "g22_corner": g22_corner,
                "g33_corner": g33_corner,
                "g_11_corner": g_11_corner,
                "g_22_corner": g_22_corner,
                "g_33_corner": g_33_corner,
                "J_corner": J_corner,
            }
        )

        # --- Calculate the curvatures ---
        bxcvx_corner, bxcvy_corner = [np.zeros(shape_corner) for _ in range(2)]

        # bxcvz = kappa / rho
        kappa_rho, kappa_z = self.magnetic_curvature(rho=rho_xy_corner, z=Z_xy_corner)
        b_rho, b_z = self.magnetic_field_direction(rho=rho_xy_corner, z=Z_xy_corner)
        bxcvz_corner = (kappa_rho * b_z - kappa_z * b_rho) / rho_xy_corner

        # Store the curvatures
        grid_data.update(
            {
                "bxcvx_corner": bxcvx_corner,
                "bxcvy_corner": bxcvy_corner,
                "bxcvz_corner": bxcvz_corner,
            }
        )

        # Interpolate the corner grids to the center, xlow, and ylow values
        grid_data_interpolated = self.interpolate_grids(grid_data)

        return grid_data_interpolated

    def generate_hermes3_mesh(self, grid_data_interpolated: dict, save_dir=None):
        """
        Generate the Hermes-3 mesh from the grid data. The grid data is expected to be in the format of
        grid_data_interpolated, which is generated by self.orthogonal_mesh() or self.cylindrical_mesh().

        Arguments:
        ---------------
        grid_data_interpolated:
            The grid data in the format of grid_data_interpolated, which contains the center, xlow, and ylow values.
        save_dir:
            The directory to save the mesh data. If None, the mesh data will not be saved.

        Return:
        ---------------
        mesh:
            An Xarray Dataset containing the mesh data in Hermes-3 format.
        """

        grid_data_hermes3 = {}

        # Clean up dimension to (nrho+4, nz)
        for key in grid_data_interpolated.keys():
            grid_data_hermes3.update(
                {key: grid_data_interpolated[key][: self.nrho + 4, : self.nz]}
            )

        # Transform data to Dataset using Xarray
        ds = xr.Dataset()
        ds["nx"] = xr.DataArray(np.array(self.nrho + 4, dtype=np.int32))
        ds["ny"] = xr.DataArray(np.array(self.nz, dtype=np.int32))
        ds['nz'] = xr.DataArray(np.array(self.ntheta, dtype=np.int32))
        ds['dz'] = xr.DataArray(np.zeros((self.nrho+4, self.nz)) + (2.*np.pi / self.ntheta))

        # ensure the regions belongs to the open-field-line region
        # NOTE: Are 0 and -1 the same?
        ds["ixseps1"] = xr.DataArray(np.array(-1, dtype=np.int32))
        ds["ixseps2"] = xr.DataArray(np.array(-1, dtype=np.int32))

        ds["jyseps1_1"] = xr.DataArray(np.array(-1, dtype=np.int32))
        ds["jyseps1_2"] = xr.DataArray(np.array(int(self.nz / 2), dtype=np.int32))
        ds["jyseps2_1"] = xr.DataArray(np.array(int(self.nz / 2), dtype=np.int32))
        ds["jyseps2_2"] = xr.DataArray(np.array(int(self.nz - 1), dtype=np.int32))

        for key in grid_data_hermes3.keys():
            ds[key] = xr.DataArray(
                data=grid_data_hermes3[key],
                dims=("x", "y"),
                attrs={"bout_type": "Field2D"},
            )

        if save_dir is not None:
            ds.to_netcdf(save_dir, mode="w", format="NETCDF4", engine="netcdf4")
            print("Mesh data saved to {}".format(save_dir))

        return ds
