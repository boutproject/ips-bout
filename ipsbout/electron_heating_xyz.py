from ipsframework import Component
from boututils.datafile import DataFile
import numpy as np

import logging

logger = logging.getLogger(__name__)


class electron_heating_xyz(Component):
    """ """

    def __init__(self, services, config):
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

    def step(self, timestamp=0.0):
        from scipy.interpolate import griddata

        data = np.loadtxt(self.HEATING_FILE, comments="%")

        zs = data[:, 2] + self.axial_offset

        with DataFile(self.GRIDFILE) as grid:
            nz = grid["nz"]
            Rxy = grid["Rxy"]
            Zxy = grid["Zxy"]
            J = grid["J"]
            dx = grid["dx"]
            dy = grid["dy"]

        dz = 2 * np.pi / nz
        dV = J * dx * dy * dz
        dV_xyz = np.repeat(dV[..., np.newaxis], nz, axis=-1)  # Volume of each cell

        thetas = np.linspace(0, 2 * np.pi, nz, endpoint=False)

        grid_r = np.repeat(Rxy[..., np.newaxis], nz, axis=-1)
        grid_z = np.repeat(Zxy[..., np.newaxis], nz, axis=-1)
        grid_theta = np.tile(thetas, Rxy.shape + (1,))

        grid_x = grid_r * np.cos(grid_theta)
        grid_y = grid_r * np.sin(grid_theta)

        ymin = np.argmin(np.abs(grid_z[0, :, 0] - np.amin(zs)))
        ymax = np.argmin(np.abs(grid_z[0, :, 0] - np.amax(zs)))

        grid_values = griddata(
            data[:, :3],
            data[:, 3],
            (
                grid_x[:, ymin:ymax, :],
                grid_y[:, ymin:ymax, :],
                grid_z[:, ymin:ymax, :] - self.axial_offset,
            ),
            method="linear",
            fill_value=0.0,
        )

        total_power = np.sum(grid_values * dV_xyz[:, ymin:ymax, :])

        print(
            f"Maximum power density of input {np.amax(data[:, 3])} -> interpolated {np.amax(grid_values)}"
        )
        print(f"Total domain volume: {np.sum(dV_xyz)} m^3")
        print(f"Total input power {total_power} W")

        grid_values *= self.total_power / total_power

        # Electron pressure source in Pascals per second. 3/2 * Pe_src is the electron
        # heating power density in Watts per cubic meter
        Pe_src = np.zeros(dV_xyz.shape)
        Pe_src[:, ymin:ymax, :] = (2.0 / 3) * grid_values

        # Write to BOUT++ grid
        # This will modify the file in-place
        with DataFile(self.GRIDFILE, write=True) as grid:
            grid["Pe_src"] = Pe_src

        print(f"Pe_src written to {self.GRIDFILE}")
