"""Rhino 8 / Grasshopper Python 3: export RC column geometry + elastic inputs.

This script exports ONE JSON configuration for an Abaqus importer.
The column axis is GLOBAL +X (0 to ColumnLength); the section is GLOBAL YZ.

GRASSHOPPER INPUTS (exact names):
    Folder             Directory to write model_config.json
    ConcreteBoundary   Closed concrete section curve:
                       * YZ plane at X=0 (preferred), OR
                       * old XY plane at Z=0 (mapped as local XY -> global YZ)
    Longitudinal       Longitudinal REBAR CENTERLINE Rhino curves in global XYZ
    Stirrups           Closed stirrup CENTERLINE Rhino curves in global XYZ
    ColumnLength       Column length along global +X
    DiaLong            Longitudinal diameter (e.g. 16 if Rhino model is in mm)
    DiaStirrup         Stirrup diameter (e.g. 8 if Rhino model is in mm)
    Ec                 Concrete Young's modulus in Pa (e.g. 30e9)
    NuC                Concrete Poisson ratio (e.g. 0.2)
    Es                 Steel Young's modulus in Pa (e.g. 200e9)
    NuS                Steel Poisson ratio (e.g. 0.3)
    Write              Boolean toggle

GRASSHOPPER OUTPUTS (exact names):
    ConfigPath         Path to saved JSON (None if Write=False)
    Status             Summary or explicit error

LENGTH UNITS: Grasshopper geometry, ColumnLength and diameters are assumed to
use the active Rhino model's length units. They are automatically converted to
METERS before writing the JSON. E inputs are ALREADY in Pa and stay in Pa.
The JSON consistently uses m, N and Pa. No mesh, BCs, loads or jobs are saved.

Important: Abaqus importer must read schema_version=2, concrete_boundary_yz
and extrude along +X. An old importer expecting XY at Z=0 must be adapted.
"""

import json
import math
import os

import Rhino
import Rhino.Geometry as rg

# Coordinate checks in output METERS, e.g. 1 micrometer.
PLANE_TOL_M = 1.0e-6
POINT_TOL_M = 1.0e-9


def unwrap(value):
    return getattr(value, "Value", value)


def list_of_curves(data):
    """Accept Item/List/Tree Access inputs, including GH_Goo and .NET lists."""
    if data is None:
        return []
    if hasattr(data, "Branches"):
        result = []
        for branch in data.Branches:
            result.extend(list_of_curves(branch))
        return result
    value = unwrap(data)
    if value is None:
        return []
    if isinstance(value, rg.Curve):
        return [value]
    if isinstance(value, (str, bytes)):
        raise TypeError("Expected Rhino Curve(s), received text.")
    try:
        iterator = iter(value)
    except TypeError:
        raise TypeError("Expected Rhino Curve(s), received: {}".format(type(value)))
    output = []
    for item in iterator:
        output.extend(list_of_curves(item))
    return output


def model_scale_to_meters():
    doc = Rhino.RhinoDoc.ActiveDoc
    if doc is None:
        raise RuntimeError("No active Rhino document: cannot determine length units.")
    source_units = doc.ModelUnitSystem
    scale = float(Rhino.RhinoMath.UnitScale(source_units, Rhino.UnitSystem.Meters))
    if not math.isfinite(scale) or scale <= 0:
        raise RuntimeError("Cannot convert model length units '{}' to meters.".format(source_units))
    return scale, str(source_units)


def point_m(p, scale):
    return [float(p.X) * scale, float(p.Y) * scale, float(p.Z) * scale]


def same_point(a, b):
    return all(abs(x - y) <= POINT_TOL_M for x, y in zip(a, b))


def curve_points_m(curve, scale, closed=False):
    curve = unwrap(curve)
    if not isinstance(curve, rg.Curve):
        raise TypeError("Expected a Rhino Curve, received {}".format(type(curve)))
    if closed and not curve.IsClosed:
        raise ValueError("ConcreteBoundary and all stirrups must be closed curves.")

    success, polyline = curve.TryGetPolyline()
    if success:
        pts = [point_m(p, scale) for p in polyline]
    else:
        # Discretization affects JSON geometry only, NOT FEM mesh size.
        # Use many points for smooth loops; straight polylines remain exact above.
        count = max(32, min(256, int(math.ceil(curve.GetLength() / max(0.01 / scale, 1e-9)))))
        params = curve.DivideByCount(count, True)
        if params is None or len(params) == 0:
            raise ValueError("Could not sample a Rhino curve.")
        pts = [point_m(curve.PointAt(t), scale) for t in params]

    unique = []
    for p in pts:
        if not all(math.isfinite(q) for q in p):
            raise ValueError("Curve contains non-finite coordinates.")
        if not unique or not same_point(unique[-1], p):
            unique.append(p)
    if len(unique) > 2 and same_point(unique[0], unique[-1]):
        unique.pop()  # Loops are implicitly closed in the JSON.
    if len(unique) < (3 if closed else 2):
        raise ValueError("Curve has too few unique vertices.")
    return unique


def area2(pts):
    return sum(pts[i][0] * pts[(i + 1) % len(pts)][1]
               - pts[(i + 1) % len(pts)][0] * pts[i][1]
               for i in range(len(pts)))


def concrete_section_yz_m(boundary, scale):
    """Return polygon in global (Y,Z) meters, retaining old XY sections."""
    points = curve_points_m(boundary, scale, closed=True)
    lies_in_yz = all(abs(p[0]) <= PLANE_TOL_M for p in points)
    lies_in_xy = all(abs(p[2]) <= PLANE_TOL_M for p in points)

    if lies_in_yz and not lies_in_xy:
        yz = [[p[1], p[2]] for p in points]
        interpretation = "YZ section at X=0"
    elif lies_in_xy and not lies_in_yz:
        yz = [[p[0], p[1]] for p in points]
        interpretation = "XY section remapped: old X->global Y, old Y->global Z"
    elif lies_in_yz and lies_in_xy:
        raise ValueError("ConcreteBoundary is degenerate (no 2D area).")
    else:
        raise ValueError(
            "ConcreteBoundary must lie at global X=0 (YZ plane) or Z=0 "
            "(XY plane, remapped). Move or rotate the curve before export.")

    if abs(area2(yz)) < 1.0e-12:
        raise ValueError("ConcreteBoundary has zero or near-zero cross-sectional area.")
    if area2(yz) < 0:
        yz.reverse()  # Consistent CCW orientation when viewed toward +X.
    return yz, interpretation


def positive(x, name):
    try:
        num = float(unwrap(x))
    except (TypeError, ValueError):
        raise ValueError("{} must be a number.".format(name))
    if not math.isfinite(num) or num <= 0:
        raise ValueError("{} must be finite and positive.".format(name))
    return num


def poisson(x, name):
    val = float(unwrap(x))
    if not math.isfinite(val) or not -1.0 < val < 0.5:
        raise ValueError("{} must satisfy -1 < nu < 0.5.".format(name))
    return val


def bool_input(x):
    value = unwrap(x)
    return value is True or str(value).strip().lower() in ("true", "1", "on", "yes")


def export_model():
    if Folder is None or not str(unwrap(Folder)).strip():
        raise ValueError("Folder is empty. Connect the JSON destination directory.")
    folder = os.path.abspath(os.path.expanduser(str(unwrap(Folder)).strip().strip('"')))

    scale, rhino_units = model_scale_to_meters()
    L = positive(ColumnLength, "ColumnLength") * scale
    d_long = positive(DiaLong, "DiaLong") * scale
    d_stirrup = positive(DiaStirrup, "DiaStirrup") * scale
    E_c = positive(Ec, "Ec")  # Pa, not MPa or GPa
    E_s = positive(Es, "Es")  # Pa, not MPa or GPa
    nu_c = poisson(NuC, "NuC")
    nu_s = poisson(NuS, "NuS")

    if L > 100.0 or L < 0.1:
        raise ValueError("ColumnLength = {:.6g} m. Check Rhino units and slider.".format(L))
    if not 0.001 <= d_long <= 0.1 or not 0.001 <= d_stirrup <= 0.1:
        raise ValueError("Rebar diameters must be 0.001..0.1 m; check model units.")
    if E_c < 1e8 or E_s < 1e8:
        raise ValueError("Ec and Es must be supplied in Pa (e.g. 30e9 and 200e9).")

    boundary = unwrap(ConcreteBoundary)
    if not isinstance(boundary, rg.Curve):
        raise ValueError("ConcreteBoundary must be ONE closed Curve (Item Access).")
    section_yz, section_method = concrete_section_yz_m(boundary, scale)
    longitudinal = [curve_points_m(c, scale) for c in list_of_curves(Longitudinal)]
    stirrups = [curve_points_m(c, scale, closed=True) for c in list_of_curves(Stirrups)]

    for name, group in (("Longitudinal", longitudinal), ("Stirrups", stirrups)):
        for index, vertices in enumerate(group, start=1):
            for p in vertices:
                if not -PLANE_TOL_M <= p[0] <= L + PLANE_TOL_M:
                    raise ValueError(
                        "{} curve {} at X={:.6g} m lies outside column X=0..{:.6g} m."
                        .format(name, index, p[0], L))

    config = {
        "schema_version": 2,
        "units": {"length": "m", "force": "N", "stress": "Pa"},
        "geometry": {
            "column_axis": "X",
            "column_origin_xyz": [0.0, 0.0, 0.0],
            "column_length": L,
            "concrete_boundary_yz": section_yz,
            "longitudinal_curves": longitudinal,
            "stirrup_curves": stirrups
        },
        "sections": {
            "longitudinal_diameter": d_long,
            "stirrup_diameter": d_stirrup
        },
        "materials": {
            "concrete": {"E": E_c, "nu": nu_c},
            "steel": {"E": E_s, "nu": nu_s}
        }
    }

    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "model_config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    status = (
        "Export completed: {}\n"
        "Rhino model units: {} (factor to m: {})\n"
        "Concrete axis: global +X, length: {:.6g} m\n"
        "Cross-section: {}\n"
        "Longitudinal curves: {} | Stirrup curves: {}\n"
        "Diameters: longitudinal {:.4g} m | stirrup {:.4g} m\n"
        "E_concrete: {:.6g} Pa | E_steel: {:.6g} Pa\n"
        "No mesh, loads, boundary conditions, steps, interactions or jobs exported.\n"
        "NOTE: Update Abaqus importer to read section YZ and extrude along +X."
    ).format(path, rhino_units, scale, L, section_method,
             len(longitudinal), len(stirrups), d_long, d_stirrup, E_c, E_s)
    return path, status


ConfigPath = None
Status = "Set Write=True to export geometry, diameters and elastic materials."
if bool_input(Write):
    try:
        ConfigPath, Status = export_model()
    except Exception as err:
        Status = "ERROR: {}".format(err)
        try:
            import Grasshopper
            ghenv.Component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Error, Status)
        except Exception:
            pass
