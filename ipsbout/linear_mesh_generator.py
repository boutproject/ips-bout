from ipsframework import Component
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)


def imas_coils(pf_active):
    """
    Extract coil information from an IMAS pf_active IDS
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
    """
    calculate which cells are outside the wall.
    returns a penalty_mask 2D array.

    Adapted from Hypnotoad
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
    """
    # BOUT++ linear mesh generator

    Generate a BOUT++ mesh for a linear device.

    ## Configuration options

    MACHINE_FILE   Machine description in JSON format.
      Contains IMAS 'pf_active' and 'wall' IDS,
      describing the coils and wall outline.

    GRIDFILE = BOUT++ grid file to be created

    The following options should be set:

    INPUT_FILES = ${MACHINE_FILE}
    OUTPUT_FILES = ${GRIDFILE}

    N_RADIAL_CELLS      # int
    N_AXIAL_CELLS       # int
    R_MIN               # float, meters
    R_MAX               # float, meters
    Z_MIN               # float, meters
    Z_MAX               # float, meters

    """

    def __init__(self, services, config):
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
            ntheta=1,
            rho_range=[self.r_min, self.r_max],
            z_range=[self.z_min, self.z_max],
            radial_distance="physical",
        )
        grid_data = AMM.orthogonal_mesh()
        AMM.generate_hermes3_mesh(grid_data, save_dir=self.GRIDFILE)

        penalty_mask = calc_penalty_mask(grid_data, wall_rz)

        with DataFile(self.GRIDFILE, write=True) as f:
            f["penalty_mask"] = penalty_mask
