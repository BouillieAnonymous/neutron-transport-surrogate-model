"""Monte Carlo tools for neutron transport simulations.

All distances are in cm. Microscopic cross-sections are read in barns and
converted to macroscopic cross-sections in cm^-1.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, exp, log, pi, sin, sqrt
from pathlib import Path
from typing import Iterable

import numpy as np


AVOGADRO = 6.02214076e23
BARN_CM2 = 1.0e-24


@dataclass(frozen=True)
class Material:
    name: str
    sigma_a_barn: float
    sigma_s_barn: float
    density_g_cm3: float
    molar_mass_g_mol: float

    @property
    def number_density(self) -> float:
        return self.density_g_cm3 * AVOGADRO / self.molar_mass_g_mol

    @property
    def macro_absorption(self) -> float:
        return self.number_density * self.sigma_a_barn * BARN_CM2

    @property
    def macro_scattering(self) -> float:
        return self.number_density * self.sigma_s_barn * BARN_CM2

    @property
    def macro_total(self) -> float:
        return self.macro_absorption + self.macro_scattering

    @property
    def total_mean_free_path(self) -> float:
        return 1.0 / self.macro_total

    @property
    def absorption_mean_free_path(self) -> float:
        return 1.0 / self.macro_absorption

    @property
    def absorption_probability(self) -> float:
        return self.macro_absorption / self.macro_total


MATERIALS = {
    "water": Material("Water", 0.6652, 103.0, 1.00, 18.0153),
    "lead": Material("Lead", 0.158, 11.221, 11.35, 207.2),
    "graphite": Material("Graphite", 0.0045, 4.74, 1.67, 12.011),
}


def uniform_diagnostic(rng: np.random.Generator, n: int = 100_000) -> dict[str, float]:
    values = rng.uniform(-2.0, 3.0, n)
    return {
        "n": float(n),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "expected_mean": 0.5,
        "variance": float(values.var(ddof=1)),
        "expected_variance": (5.0**2) / 12.0,
    }


def random_points_3d(
    rng: np.random.Generator, n: int = 2_000, low: float = 0.0, high: float = 1.0
) -> np.ndarray:
    return rng.uniform(low, high, size=(n, 3))


def bad_lcg_points(n: int = 2_000) -> np.ndarray:
    """Small-modulus LCG resembling the spectral-problem demonstration."""

    modulus = 2**16
    a = 137
    c = 187
    x = 1
    values = np.empty(3 * n)
    for i in range(3 * n):
        x = (a * x + c) % modulus
        values[i] = x / modulus
    return values.reshape(n, 3)


def exponential_samples(
    rng: np.random.Generator, mean_free_path: float, n: int = 100_000
) -> np.ndarray:
    return rng.exponential(mean_free_path, size=n)


def isotropic_unit_vectors(rng: np.random.Generator, n: int) -> np.ndarray:
    phi = rng.uniform(0.0, 2.0 * pi, n)
    cos_theta = rng.uniform(-1.0, 1.0, n)
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    return np.column_stack((sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta))


def isotropic_steps(rng: np.random.Generator, mean_free_path: float, n: int) -> np.ndarray:
    lengths = exponential_samples(rng, mean_free_path, n)
    return isotropic_unit_vectors(rng, n) * lengths[:, None]


def weighted_linear_fit(x: np.ndarray, y: np.ndarray, sigma_y: np.ndarray) -> tuple[float, float, float, float]:
    """Return slope, intercept, slope_err, intercept_err for y = slope*x + intercept."""

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(sigma_y) & (sigma_y > 0)
    x = x[mask]
    y = y[mask]
    sigma_y = sigma_y[mask]
    if len(x) < 3:
        raise ValueError("at least three valid points are required for a weighted fit")

    weights = 1.0 / sigma_y**2
    design = np.column_stack((x, np.ones_like(x)))
    normal = design.T @ (weights[:, None] * design)
    cov = np.linalg.inv(normal)
    beta = cov @ (design.T @ (weights * y))
    return float(beta[0]), float(beta[1]), float(sqrt(cov[0, 0])), float(sqrt(cov[1, 1]))


def fit_exponential_length_from_samples(
    samples: np.ndarray, bins: int = 70, min_count: int = 15
) -> dict[str, float]:
    counts, edges = np.histogram(samples, bins=bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    mask = counts >= min_count
    y = np.log(counts[mask])
    sigma_y = 1.0 / np.sqrt(counts[mask])
    slope, intercept, slope_err, intercept_err = weighted_linear_fit(centres[mask], y, sigma_y)
    length = -1.0 / slope
    length_err = slope_err / slope**2
    return {
        "length": length,
        "length_err": length_err,
        "slope": slope,
        "slope_err": slope_err,
        "intercept": intercept,
        "intercept_err": intercept_err,
    }


def trace_neutron(
    rng: np.random.Generator,
    material: Material,
    thickness: float,
    max_steps: int = 20_000,
) -> tuple[str, np.ndarray]:
    position = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    path = [position.copy()]

    for _ in range(max_steps):
        step_length = rng.exponential(material.total_mean_free_path)
        next_position = position + step_length * direction
        path.append(next_position.copy())

        if next_position[0] < 0.0:
            return "reflected", np.array(path)
        if next_position[0] > thickness:
            return "transmitted", np.array(path)
        if rng.uniform() < material.absorption_probability:
            return "absorbed", np.array(path)

        position = next_position
        direction = isotropic_unit_vectors(rng, 1)[0]

    return "max_steps", np.array(path)


def simulate_slab(
    rng: np.random.Generator,
    material: Material,
    thickness: float,
    n_neutrons: int,
    max_steps: int = 20_000,
) -> dict[str, float]:
    counts = {"absorbed": 0, "reflected": 0, "transmitted": 0, "max_steps": 0}
    for _ in range(n_neutrons):
        outcome, _ = trace_neutron(rng, material, thickness, max_steps=max_steps)
        counts[outcome] += 1

    result: dict[str, float] = {"thickness_cm": float(thickness), "n": float(n_neutrons)}
    for outcome in ("absorbed", "reflected", "transmitted", "max_steps"):
        count = counts[outcome]
        fraction = count / n_neutrons
        uncertainty = sqrt(fraction * (1.0 - fraction) / n_neutrons)
        result[f"{outcome}_count"] = float(count)
        result[f"{outcome}_fraction"] = fraction
        result[f"{outcome}_uncertainty"] = uncertainty
    return result


def scan_thicknesses(
    rng: np.random.Generator,
    material: Material,
    thicknesses: Iterable[float],
    n_neutrons: int,
) -> list[dict[str, float]]:
    return [simulate_slab(rng, material, float(l), n_neutrons) for l in thicknesses]


def fit_attenuation_from_transmission(rows: list[dict[str, float]]) -> dict[str, float]:
    thickness = np.array([row["thickness_cm"] for row in rows], dtype=float)
    transmission = np.array([row["transmitted_fraction"] for row in rows], dtype=float)
    sigma_t = np.array([row["transmitted_uncertainty"] for row in rows], dtype=float)
    mask = (transmission > 0.0) & (sigma_t > 0.0)

    log_t = np.log(transmission[mask])
    sigma_log_t = sigma_t[mask] / transmission[mask]
    slope, intercept, slope_err, intercept_err = weighted_linear_fit(
        thickness[mask], log_t, sigma_log_t
    )
    attenuation_length = -1.0 / slope
    attenuation_length_err = slope_err / slope**2
    return {
        "attenuation_length_cm": attenuation_length,
        "attenuation_length_err_cm": attenuation_length_err,
        "slope": slope,
        "slope_err": slope_err,
        "intercept": intercept,
        "intercept_err": intercept_err,
        "valid_points": float(mask.sum()),
    }


def simulate_two_slab_woodcock(
    rng: np.random.Generator,
    left: Material,
    right: Material,
    left_thickness: float = 10.0,
    right_thickness: float = 10.0,
    n_neutrons: int = 10_000,
    max_steps: int = 50_000,
) -> dict[str, float]:
    total_thickness = left_thickness + right_thickness
    majorant = max(left.macro_total, right.macro_total)
    counts = {"absorbed": 0, "reflected": 0, "transmitted": 0, "max_steps": 0}

    for _ in range(n_neutrons):
        position = np.array([0.0, 0.0, 0.0])
        direction = np.array([1.0, 0.0, 0.0])

        for _step in range(max_steps):
            position = position + rng.exponential(1.0 / majorant) * direction
            x = position[0]

            if x < 0.0:
                counts["reflected"] += 1
                break
            if x > total_thickness:
                counts["transmitted"] += 1
                break

            material = left if x <= left_thickness else right
            if rng.uniform() > material.macro_total / majorant:
                continue
            if rng.uniform() < material.absorption_probability:
                counts["absorbed"] += 1
                break
            direction = isotropic_unit_vectors(rng, 1)[0]
        else:
            counts["max_steps"] += 1

    result: dict[str, float] = {"n": float(n_neutrons)}
    for outcome, count in counts.items():
        fraction = count / n_neutrons
        result[f"{outcome}_fraction"] = fraction
        result[f"{outcome}_uncertainty"] = sqrt(fraction * (1.0 - fraction) / n_neutrons)
        result[f"{outcome}_count"] = float(count)
    return result


def _svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
    ]


def save_line_svg(
    path: Path,
    series: list[tuple[str, np.ndarray, np.ndarray, str]],
    title: str,
    x_label: str,
    y_label: str,
    width: int = 900,
    height: int = 560,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    left, right, top, bottom = 78, 24, 54, 70
    all_x = np.concatenate([item[1] for item in series])
    all_y = np.concatenate([item[2] for item in series])
    x_min, x_max = float(all_x.min()), float(all_x.max())
    y_min, y_max = float(min(0.0, all_y.min())), float(max(1.0, all_y.max()))
    if x_min == x_max:
        x_max += 1.0
    if y_min == y_max:
        y_max += 1.0

    def sx(x: float) -> float:
        return left + (x - x_min) * (width - left - right) / (x_max - x_min)

    def sy(y: float) -> float:
        return height - bottom - (y - y_min) * (height - top - bottom) / (y_max - y_min)

    parts = _svg_header(width, height)
    parts.append(f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="22" font-family="Arial">{title}</text>')
    parts.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#252525"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#252525"/>')

    for frac in np.linspace(0.0, 1.0, 6):
        x_val = x_min + frac * (x_max - x_min)
        px = sx(x_val)
        parts.append(f'<line x1="{px:.1f}" y1="{height-bottom}" x2="{px:.1f}" y2="{height-bottom+5}" stroke="#252525"/>')
        parts.append(f'<text x="{px:.1f}" y="{height-bottom+24}" text-anchor="middle" font-size="12" font-family="Arial">{x_val:.0f}</text>')
        y_val = y_min + frac * (y_max - y_min)
        py = sy(y_val)
        parts.append(f'<line x1="{left-5}" y1="{py:.1f}" x2="{left}" y2="{py:.1f}" stroke="#252525"/>')
        parts.append(f'<text x="{left-10}" y="{py+4:.1f}" text-anchor="end" font-size="12" font-family="Arial">{y_val:.2f}</text>')

    for name, xs, ys, color in series:
        points = " ".join(f"{sx(float(x)):.1f},{sy(float(y)):.1f}" for x, y in zip(xs, ys))
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.6" points="{points}"/>')
        for x, y in zip(xs, ys):
            parts.append(f'<circle cx="{sx(float(x)):.1f}" cy="{sy(float(y)):.1f}" r="3.2" fill="{color}"/>')

    parts.append(f'<text x="{width/2}" y="{height-22}" text-anchor="middle" font-size="15" font-family="Arial">{x_label}</text>')
    parts.append(f'<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle" font-size="15" font-family="Arial">{y_label}</text>')
    for i, (name, _xs, _ys, color) in enumerate(series):
        y = top + 24 * i
        parts.append(f'<line x1="{width-190}" y1="{y}" x2="{width-160}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{width-152}" y="{y+5}" font-size="14" font-family="Arial">{name}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def save_scatter3d_svg(
    path: Path,
    points: np.ndarray,
    title: str,
    color: str = "#2563eb",
    width: int = 700,
    height: int = 620,
    max_points: int = 2_000,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = points[:max_points]
    centred = pts - pts.mean(axis=0)
    scale = np.max(np.abs(centred))
    if scale == 0.0:
        scale = 1.0
    centred = centred / scale
    px = centred[:, 0] * 0.86 - centred[:, 1] * 0.86
    py = centred[:, 0] * 0.38 + centred[:, 1] * 0.38 - centred[:, 2] * 0.86
    px = width / 2 + px * width * 0.34
    py = height / 2 + py * height * 0.34

    parts = _svg_header(width, height)
    parts.append(f'<text x="{width/2}" y="32" text-anchor="middle" font-size="22" font-family="Arial">{title}</text>')
    order = np.argsort(centred[:, 2])
    for i in order:
        opacity = 0.35 + 0.45 * (centred[i, 2] + 1.0) / 2.0
        parts.append(f'<circle cx="{px[i]:.1f}" cy="{py[i]:.1f}" r="2.0" fill="{color}" opacity="{opacity:.2f}"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def save_path_svg(path: Path, paths: list[np.ndarray], title: str, thickness: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 860, 480
    left, right, top, bottom = 70, 28, 48, 54
    all_points = np.vstack(paths)
    y_extent = max(1.0, float(np.max(np.abs(all_points[:, 1]))))
    x_min, x_max = -0.05 * thickness, 1.05 * thickness

    def sx(x: float) -> float:
        return left + (x - x_min) * (width - left - right) / (x_max - x_min)

    def sy(y: float) -> float:
        return height / 2 - y * (height - top - bottom) / (2.3 * y_extent)

    colors = ["#2563eb", "#b45309", "#047857", "#be123c", "#6d28d9", "#0f766e"]
    parts = _svg_header(width, height)
    parts.append(f'<text x="{width/2}" y="28" text-anchor="middle" font-size="22" font-family="Arial">{title}</text>')
    parts.append(f'<rect x="{sx(0):.1f}" y="{top}" width="{sx(thickness)-sx(0):.1f}" height="{height-top-bottom}" fill="#e8eef8" opacity="0.65"/>')
    parts.append(f'<line x1="{sx(0):.1f}" y1="{top}" x2="{sx(0):.1f}" y2="{height-bottom}" stroke="#111827" stroke-width="2"/>')
    parts.append(f'<line x1="{sx(thickness):.1f}" y1="{top}" x2="{sx(thickness):.1f}" y2="{height-bottom}" stroke="#111827" stroke-width="2"/>')
    for idx, pts in enumerate(paths):
        points = " ".join(f"{sx(float(p[0])):.1f},{sy(float(p[1])):.1f}" for p in pts)
        color = colors[idx % len(colors)]
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.8" opacity="0.88" points="{points}"/>')
        parts.append(f'<circle cx="{sx(float(pts[-1,0])):.1f}" cy="{sy(float(pts[-1,1])):.1f}" r="3.5" fill="{color}"/>')
    parts.append(f'<text x="{sx(0):.1f}" y="{height-20}" text-anchor="middle" font-size="13" font-family="Arial">x=0</text>')
    parts.append(f'<text x="{sx(thickness):.1f}" y="{height-20}" text-anchor="middle" font-size="13" font-family="Arial">x=L</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
