"""Run the neutron transport simulation and generate results.

Usage:
    python scripts/run_neutron_transport.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neutron_transport import (
    MATERIALS,
    exponential_samples,
    fit_attenuation_from_transmission,
    fit_exponential_length_from_samples,
    isotropic_steps,
    isotropic_unit_vectors,
    random_points_3d,
    bad_lcg_points,
    save_line_svg,
    save_path_svg,
    save_scatter3d_svg,
    scan_thicknesses,
    simulate_slab,
    simulate_two_slab_woodcock,
    trace_neutron,
    uniform_diagnostic,
)


OUT = ROOT / "results"


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def material_property_rows() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for key, material in MATERIALS.items():
        rows.append(
            {
                "material": material.name,
                "number_density_cm^-3": material.number_density,
                "macro_absorption_cm^-1": material.macro_absorption,
                "macro_scattering_cm^-1": material.macro_scattering,
                "macro_total_cm^-1": material.macro_total,
                "total_mean_free_path_cm": material.total_mean_free_path,
                "absorption_mean_free_path_cm": material.absorption_mean_free_path,
                "collision_absorption_probability": material.absorption_probability,
            }
        )
    return rows


def main() -> None:
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20762)

    diagnostics = {"uniform": uniform_diagnostic(rng)}

    good_points = random_points_3d(rng, n=2_000)
    bad_points = bad_lcg_points(n=2_000)
    save_scatter3d_svg(OUT / "uniform_3d_points.svg", good_points, "numpy.random.uniform 3D points", "#2563eb")
    save_scatter3d_svg(OUT / "bad_lcg_spectral_points.svg", bad_points, "Small LCG 3D points: visible planes", "#be123c")

    water = MATERIALS["water"]
    absorption_samples = exponential_samples(rng, water.absorption_mean_free_path, n=120_000)
    absorption_fit = fit_exponential_length_from_samples(absorption_samples)
    diagnostics["water_no_scattering_attenuation"] = absorption_fit

    unit_vectors = isotropic_unit_vectors(rng, n=2_500)
    save_scatter3d_svg(OUT / "isotropic_unit_vectors.svg", unit_vectors, "Isotropic unit vectors", "#047857")

    steps = isotropic_steps(rng, water.total_mean_free_path, n=2_500)
    save_scatter3d_svg(OUT / "isotropic_exponential_steps_water.svg", steps, "Isotropic exponential steps in water", "#7c3aed")

    props = material_property_rows()
    write_csv(OUT / "material_properties.csv", props)

    thickness = 10.0
    slab_rows: list[dict[str, float | str]] = []
    random_walk_paths: dict[str, list[np.ndarray]] = {}
    for key, material in MATERIALS.items():
        paths = []
        for _ in range(6):
            _outcome, path = trace_neutron(rng, material, thickness)
            paths.append(path)
        random_walk_paths[key] = paths
        save_path_svg(OUT / f"random_walk_{key}.svg", paths, f"{material.name}: sample neutron paths", thickness)

        row = simulate_slab(rng, material, thickness, n_neutrons=6_000)
        row["material"] = material.name
        slab_rows.append(row)
    write_csv(OUT / "slab_10cm_rates.csv", slab_rows)

    thicknesses = np.arange(1.0, 31.0, 2.0)
    scan_rows: list[dict[str, float | str]] = []
    fit_rows: list[dict[str, float | str]] = []
    colors = {"water": "#2563eb", "lead": "#b45309", "graphite": "#047857"}
    transmission_series = []
    reflection_series = []
    absorption_series = []

    for key, material in MATERIALS.items():
        rows = scan_thicknesses(rng, material, thicknesses, n_neutrons=3_000)
        for row in rows:
            row["material"] = material.name
            scan_rows.append(row)

        fit = fit_attenuation_from_transmission(rows)
        fit["material"] = material.name
        fit_rows.append(fit)

        xs = np.array([row["thickness_cm"] for row in rows], dtype=float)
        transmission_series.append((material.name, xs, np.array([row["transmitted_fraction"] for row in rows]), colors[key]))
        reflection_series.append((material.name, xs, np.array([row["reflected_fraction"] for row in rows]), colors[key]))
        absorption_series.append((material.name, xs, np.array([row["absorbed_fraction"] for row in rows]), colors[key]))

    write_csv(OUT / "thickness_scan_rates.csv", scan_rows)
    write_csv(OUT / "attenuation_lengths.csv", fit_rows)
    save_line_svg(OUT / "transmission_vs_thickness.svg", transmission_series, "Transmission vs slab thickness", "Thickness L (cm)", "Transmission fraction")
    save_line_svg(OUT / "reflection_vs_thickness.svg", reflection_series, "Reflection vs slab thickness", "Thickness L (cm)", "Reflection fraction")
    save_line_svg(OUT / "absorption_vs_thickness.svg", absorption_series, "Absorption vs slab thickness", "Thickness L (cm)", "Absorption fraction")

    woodcock_rows: list[dict[str, float | str]] = []
    woodcock_cases = [
        ("Water + Lead", MATERIALS["water"], MATERIALS["lead"]),
        ("Lead + Graphite", MATERIALS["lead"], MATERIALS["graphite"]),
        ("Water + Graphite", MATERIALS["water"], MATERIALS["graphite"]),
    ]
    for label, left, right in woodcock_cases:
        row = simulate_two_slab_woodcock(rng, left, right, n_neutrons=4_000)
        row["case"] = label
        woodcock_rows.append(row)
    write_csv(OUT / "woodcock_two_slab_rates.csv", woodcock_rows)

    diagnostics["runtime_seconds"] = time.perf_counter() - start
    (OUT / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    print(f"Results written to {OUT}")
    print(f"Runtime: {diagnostics['runtime_seconds']:.2f} s")
    print(f"Water no-scattering attenuation length: {absorption_fit['length']:.2f} +/- {absorption_fit['length_err']:.2f} cm")
    print("10 cm slab rates:")
    for row in slab_rows:
        print(
            f"  {row['material']}: A={row['absorbed_fraction']:.3f}, "
            f"R={row['reflected_fraction']:.3f}, T={row['transmitted_fraction']:.3f}"
        )
    print("Transmission attenuation lengths:")
    for row in fit_rows:
        print(
            f"  {row['material']}: {row['attenuation_length_cm']:.2f} "
            f"+/- {row['attenuation_length_err_cm']:.2f} cm"
        )


if __name__ == "__main__":
    main()
