# Gather Hermes-3 linear device simulation outputs
#
from boutdata import collect
from boututils.datafile import DataFile
import numpy as np
import os

paths = [
    ("11-it4", [0, -1]),
]
gridfilepath = "ProtoMPEX_29128_x68y64z64_penalty_helicon_it3.nc"

with DataFile(gridfilepath) as f:
    R = np.array(f["Rxy"][2:-2, :])
    Z = np.array(f["Zxy"][2:-2, :])

path0 = paths[0][0]
path_1 = paths[-1][0]

wci = collect("Omega_ci", path=path0, info=False)
cs0 = collect("Cs0", path=path0, info=False)
rho_s0 = cs0 / wci
Nnorm = collect("Nnorm", path=path0, info=False)
Tnorm = collect("Tnorm", path=path0, info=False)


def collect_all(varname):
    return np.concatenate(
        [
            collect(varname, path=path, info=False, tind=tind, yguards=True)
            for path, tind in paths
        ],
        axis=0,
    )


def replace_yguards(var):
    result = var[:, :, 2:-2, :]
    result[:, :, 0, :] = 0.5 * (var[:, :, 1, :] + var[:, :, 2, :])
    result[:, :, -1, :] = 0.5 * (var[:, :, -2, :] + var[:, :, -3, :])
    return result


def with_attributes(var, attrs):
    var.attributes = attrs
    return var


Ne = with_attributes(
    replace_yguards(collect_all("Ne")[:, 2:-2, :, :]) * Nnorm,  # in m^-3
    {
        "long_name": "e number density",
        "short_name": "density",
        "units": "m^-3",
        "bout_type": "Field3D",
    },
)
Ne_floor = np.clip(Ne, 1e-5 * Nnorm, None)
Te = with_attributes(
    replace_yguards(collect_all("Te")[:, 2:-2, :, :]) * Tnorm,  # in eV
    {
        "long_name": "e temperature",
        "short_name": "temperature",
        "units": "eV",
        "bout_type": "Field3D",
    },
)
Ve = with_attributes(
    replace_yguards(collect_all("Ve")[:, 2:-2, :, :]) * cs0,  # in m/s
    {
        "long_name": "e parallel velocity",
        "short_name": "electron velocity",
        "units": "m/s",
        "bout_type": "Field3D",
    },
)
Ti = with_attributes(
    replace_yguards(collect_all("Pd+")[:, 2:-2, :, :])
    * (Tnorm * Nnorm)
    / Ne_floor,  # in eV
    {
        "long_name": "d+ temperature",
        "short_name": "temperature",
        "units": "eV",
        "bout_type": "Field3D",
    },
)
phi = with_attributes(
    replace_yguards(collect_all("phi")[:, 2:-2, :, :]) * Tnorm,  # V
    {
        "long_name": "plasma potential",
        "short_name": "potential",
        "units": "V",
        "bout_type": "Field3D",
    },
)
Vi = with_attributes(
    replace_yguards(collect_all("NVd+")[:, 2:-2, :, :])
    * Nnorm
    * cs0
    / (2.0 * Ne_floor),  # m/s
    {
        "long_name": "d+ parallel velocity",
        "short_name": "velocity",
        "units": "m/s",
        "bout_type": "Field3D",
    },
)
t_array = collect_all("t_array") / wci * 1e6  # in microseconds
Nd = (
    collect("Nd", path=path0, info=False)[2:-2, :, :] * Nnorm
)  # m^-3. Not time dependent
Nd.attributes = {
    "long_name": "d number density",
    "short_name": "density",
    "units": "m^-3",
    "bout_type": "Field3D",
}

nt, nx, ny, nz = Ne.shape
angle = 2 * np.pi * np.arange(nz) / nz

#########################
# Coordinates
J = collect("J", path=path0, info=False)[2:-2, :]
g_22 = collect("g_22", path=path0, info=False)[2:-2, :]
dx = collect("dx", path=path0, info=False)[2:-2, :]
dy = collect("dy", path=path0, info=False)[2:-2, :]
dz = collect("dz", path=path0, info=False)[2:-2, :]

dA = with_attributes(
    J * dx * dz / np.sqrt(g_22) * rho_s0**2,
    {
        "long_name": "sheath area",
        "short_name": "area",
        "units": "m^2",
        "bout_type": "FieldPerp",
    },
)

dV = with_attributes(
    J * dx * dy * dz * rho_s0**3,
    {
        "long_name": "cell volume",
        "short_name": "volume",
        "units": "m^3",
        "bout_type": "Field3D",
    },
)


#########################
# Sources and sinks

PowerDensityNorm = wci * 1.602e-19 * Nnorm * Tnorm

energy_sources = {}

try:
    # Electron energy sink due to excitation
    energy_sources["excitation"] = with_attributes(
        collect_all("Rd+_ex")[:, 2:-2, 2:-2, :] * PowerDensityNorm,
        {
            "long_name": "excitation radiation",
            "short_name": "R_ex",
            "units": "W m^-3",
            "bout_type": "Field3D",
        },
    )
except:
    print("No excitation radiation 'Rd+_ex'")

try:
    energy_sources["recombination"] = with_attributes(
        collect_all("Rd+_rec")[:, 2:-2, 2:-2, :] * PowerDensityNorm,
        {
            "long_name": "recombination radiation",
            "short_name": "Rd+_rec",
            "units": "W m^-3",
            "bout_type": "Field3D",
        },
    )
except:
    print("No recombination 'Rd+_rec'")

particle_sources = {}
try:
    particle_sources["ionization"] = with_attributes(
        collect_all("Sd+_iz")[:, 2:-2, 2:-2, :] * (wci * Nnorm),
        {
            "long_name": "ionization ion source",
            "short_name": "Sd+_iz",
            "units": "m^-3 s^-1",
            "bout_type": "Field3D",
        },
    )
except:
    print("No ionization 'Sd+_iz'")

try:
    particle_sources["recombination"] = with_attributes(
        collect_all("Sd+_rec")[:, 2:-2, 2:-2, :] * (wci * Nnorm),
        {
            "long_name": "recombination ion source",
            "short_name": "Sd+_rec",
            "units": "m^-3 s^-1",
            "bout_type": "Field3D",
        },
    )
except:
    print("No ionization 'Sd+_rec'")


#########################
# Penalty mask

try:
    # Penalty mask isn't time-dependent, so no t axis
    penalty_mask = with_attributes(
        collect_all("penalty_mask")[2:-2, 2:-2, :],
        {"long_name": "penalty mask", "short_name": "mask", "bout_type": "Field3D"},
    )

    # Apply mask to the particle sources
    for key, var in particle_sources.items():
        particle_sources[key] = var * penalty_mask

    # Apply mask to the energy sources
    for key, var in energy_sources.items():
        energy_sources[key] = var * penalty_mask

    # Read particle and energy sinks due to the penalty terms
    # These terms should not be multiplied by the penalty mask
    # These are only saved if diagnostics were enabled in the penalty boundary condition.

    particle_sources["penalty"] = with_attributes(
        collect_all("Sd+_penalty")[:, 2:-2, 2:-2, :] * (wci * Nnorm),
        {
            "long_name": "penalty particle sink",
            "short_name": "Sd+_penalty",
            "units": "m^-3 s^-1",
            "bout_type": "Field3D",
        },
    )

    energy_sources["ion_penalty"] = with_attributes(
        collect_all("Rd+_penalty")[:, 2:-2, 2:-2, :] * PowerDensityNorm,
        {
            "long_name": "penalty ion heat",
            "short_name": "Rd+_penalty",
            "units": "W m^-3",
            "bout_type": "Field3D",
        },
    )

    energy_sources["electron_penalty"] = with_attributes(
        collect_all("Re_penalty")[:, 2:-2, 2:-2, :] * PowerDensityNorm,
        {
            "long_name": "penalty electron heat",
            "short_name": "Re_penalty",
            "units": "W m^-3",
            "bout_type": "Field3D",
        },
    )
except:
    print("Missing penalty variables")

# Heating. This should not be multiplied by penalty_mask
energy_sources["electron_heating_Wm-3"] = with_attributes(
    collect_all("Pe_src")[:, 2:-2, 2:-2, :]
    * ((3.0 / 2) * PowerDensityNorm),  # Pascals / second
    {
        "long_name": "electron heating",
        "short_name": "heating",
        "units": "W m^-3",
        "bout_type": "Field3D",
    },
)

#########################
# Sheaths

phi_wall = 0.0
Ge = 0.0
Zi = 1.0
AA = 2.0
qe = 1.602e-19
ion_adiabatic = 5.0 / 3

sheath_vars = {}  # [t, x, z] variables

for boundary_name, indx, direction in [("Left", 0, -1.0), ("Right", -1, 1.0)]:
    # Cell area
    area = np.repeat((dA[:, indx])[..., np.newaxis], nz, axis=-1)

    nesheath = Ne[:, :, indx, :]
    tesheath = np.clip(Te[:, :, indx, :], 1e-5, None)
    tisheath = np.clip(Ti[:, :, indx, :], 1e-5, None)
    phisheath = phi[:, :, indx, :]
    vesheath = Ve[:, :, indx, :]
    visheath = Vi[:, :, indx, :]

    gamma_e = with_attributes(
        np.clip(
            (2 / (1.0 - Ge)) + ((phisheath - phi_wall) / np.clip(tesheath, 1e-5, None)),
            0.0,
            None,
        ),
        {
            "long_name": "electron sheath heat transmission coefficient",
            "short_name": "gamma_e",
            "bout_type": "FieldPerp",
        },
    )

    electron_heat_flux = with_attributes(
        gamma_e * 1.602e-19 * tesheath * nesheath * vesheath * direction,
        {
            "long_name": "electron sheath heat flux",
            "short_name": "q_e",
            "units": "W m^-2",
            "bout_type": "FieldPerp",
        },
    )

    C_i_sq_norm = np.clip(
        (ion_adiabatic * tisheath + Zi * tesheath) / (AA * Tnorm), 0.0, 100.0
    )

    gamma_i = with_attributes(
        2.5 + (0.5 * AA * Tnorm * C_i_sq_norm / tisheath),
        {
            "long_name": "ion sheath heat transmission coefficient",
            "short_name": "gamma_i",
            "bout_type": "FieldPerp",
        },
    )

    ion_heat_flux = with_attributes(
        gamma_i * 1.602e-19 * tisheath * nesheath * visheath * direction,
        {
            "long_name": "ion sheath heat flux",
            "short_name": "q_i",
            "units": "W m^-2",
            "bout_type": "FieldPerp",
        },
    )

    ion_particle_flux = with_attributes(
        nesheath * visheath * direction,
        {
            "long_name": "ion particle flux",
            "short_name": "gamma_i",
            "units": "particles s^-1 m^-2",
            "bout_type": "FieldPerp",
        },
    )

    sheath_vars[boundary_name] = {
        "area": area,
        "electron_gamma": gamma_e,
        "electron_heat_flux": electron_heat_flux,
        "ion_gamma": gamma_i,
        "ion_heat_flux": ion_heat_flux,
        "ion_particle_flux": ion_particle_flux,
    }


Ne_av = np.mean(Ne, axis=0)
Te_av = np.mean(Te, axis=0)
Ti_av = np.mean(Ti, axis=0)
phi_av = np.mean(phi, axis=0)
Vi_av = np.mean(Vi, axis=0)

# Expand the R, Z and angle arrays to [nx, ny, nz]
R = np.repeat(R[..., np.newaxis], nz, axis=-1)
Z = np.repeat(Z[..., np.newaxis], nz, axis=-1)
angle = np.repeat(
    np.repeat(angle[np.newaxis, ...], ny, axis=0)[np.newaxis, ...], nx, axis=0
)

with DataFile(os.path.join(path_1, "time_average.nc"), create=True) as f:
    f["r"] = R
    f["z"] = Z
    f["angle"] = angle
    f["cell_volume"] = dV
    f["electron_density_m-3"] = Ne_av
    f["electron_temperature_eV"] = Te_av
    f["ion_temperature_eV"] = Ti_av
    f["plasma_potential_V"] = phi_av
    f["ion_velocity_mps"] = Vi_av
    f["neutral_density_m-3"] = Nd
    f["average_time_us"] = t_array[-1] - t_array[0]
    f["simulation_time_us"] = t_array[-1]

    for key, val in particle_sources.items():
        f[val.attributes["short_name"]] = np.mean(val, axis=0)

    for key, val in energy_sources.items():
        f[val.attributes["short_name"]] = np.mean(val, axis=0)

    for bndry_name, vals in sheath_vars.items():
        print(f"Boundary: {bndry_name}")
        for var_name, var in vals.items():
            name = bndry_name + "_" + var_name
            if var.ndim == 3:
                f[name] = np.mean(var, axis=0)
            else:
                f[name] = var
        particle_flux = with_attributes(
            np.sum(vals["area"] * np.mean(vals["ion_particle_flux"], axis=0)),
            {
                "long_name": "ion particle flow",
                "short_name": "Gamma_i",
                "units": "particles s^-1",
                "bout_type": "scalar",
            },
        )
        f[bndry_name + "_total_ion_particles"] = particle_flux
        print(f"  Particle flux: {particle_flux} /s")
        print(f"  Surface recombination: {particle_flux * 1.602e-19 * 13.6} W")

        f[bndry_name + "_total_ion_heat"] = with_attributes(
            np.sum(vals["area"] * np.mean(vals["ion_heat_flux"], axis=0)),
            {
                "long_name": "ion heat flow",
                "short_name": "Q_i",
                "units": "W",
                "bout_type": "scalar",
            },
        )
        print(f"  Ion heat: {f[bndry_name + '_total_ion_heat']} W")
        f[bndry_name + "_total_electron_heat"] = with_attributes(
            np.sum(vals["area"] * np.mean(vals["electron_heat_flux"], axis=0)),
            {
                "long_name": "electron heat flow",
                "short_name": "Q_e",
                "units": "W",
                "bout_type": "scalar",
            },
        )
        print(f"  Electron heat: {f[bndry_name + '_total_electron_heat']} W")


for tind in [0, -1]:
    with DataFile(
        os.path.join(path_1, f"snapshot_t{int(t_array[tind]):05d}.nc"), create=True
    ) as f:
        f["r"] = R
        f["z"] = Z
        f["angle"] = angle
        f["electron_density_m-3"] = Ne[tind, :, :, :]
        f["electron_temperature_eV"] = Te[tind, :, :, :]
        f["ion_temperature_eV"] = Ti[tind, :, :, :]
        f["plasma_potential_V"] = phi[tind, :, :, :]
        f["ion_velocity_mps"] = Vi[tind, :, :, :]
        f["neutral_density_m-3"] = Nd
        f["average_time_us"] = 0.0
        f["simulation_time_us"] = t_array[tind]

        for key, val in particle_sources.items():
            f[val.attributes["short_name"]] = val[tind, ...]

        for key, val in energy_sources.items():
            f[val.attributes["short_name"]] = val[tind, ...]

        for bndry_name, vals in sheath_vars.items():
            for var_name, var in vals.items():
                name = bndry_name + "_" + var_name
                if var.ndim == 3:
                    f[name] = var[tind, :, :]
                else:
                    f[name] = var
            f[bndry_name + "_total_ion_particles"] = with_attributes(
                np.sum(vals["area"] * vals["ion_particle_flux"][tind, :, :]),
                {
                    "long_name": "ion particle flow",
                    "short_name": "Gamma_i",
                    "units": "particles s^-1",
                    "bout_type": "scalar",
                },
            )
            f[bndry_name + "_total_ion_heat"] = with_attributes(
                np.sum(vals["area"] * vals["ion_heat_flux"][tind, :, :]),
                {
                    "long_name": "ion heat flow",
                    "short_name": "Q_i",
                    "units": "W",
                    "bout_type": "scalar",
                },
            )
            f[bndry_name + "_total_electron_heat"] = with_attributes(
                np.sum(vals["area"] * vals["electron_heat_flux"][tind, :, :]),
                {
                    "long_name": "electron heat flow",
                    "short_name": "Q_e",
                    "units": "W",
                    "bout_type": "scalar",
                },
            )
