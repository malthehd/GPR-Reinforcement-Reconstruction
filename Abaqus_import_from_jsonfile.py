"""Abaqus/CAE > File > Run Script.

Imports model_config.json from the Rhino 8 / Grasshopper X-axis SI exporter.

FIX: Do not pass generator expressions to Abaqus-overridden sum().
     Explicit XYZ distance calculations avoid the reported TypeError.

Creates:
  - Solid concrete (global YZ section extruded along GLOBAL +X in assembly)
  - Longitudinal Ø16 rebar and stirrup Ø8 rebar as 3D wire centerlines
  - Linear elastic concrete/steel materials, and solid/truss sections
  - Assembly instances and a saved .cae database

No mesh seeds, element types, mesh, steps, loads, supports, BCs,
interactions/embedded regions, reference points, couplings or jobs.

JSON units must be SI: m, N, Pa. E moduli are read in Pa, and
truss cross-sectional areas are created in m^2.

IMPORTANT: Abaqus BaseSolidExtrude runs along the PART's local +Z axis.
The Concrete instance is rotated +120 degrees around (1,1,1) so:
   part (sketch-X, sketch-Y, extrude-Z) -> assembly (Y, Z, X)
All rebar coordinates come from Grasshopper already in GLOBAL XYZ and are
NOT rotated again. View the ASSEMBLY (not the concrete part's local view)
for the column aligned with global +X.

Uses schema_version=2, compatible with Grasshopper_export_Abaqus_X_axis_SI.py.
Abaqus API must run inside Abaqus/CAE (not standard Python).
"""

from abaqus import mdb, getInput
from abaqusConstants import THREE_D, DEFORMABLE_BODY, ON, SEPARATE, CARTESIAN
import os
import sys
import json
import math

# Optional: Set the FULL path to skip the path popup in Abaqus/CAE.
# Example: CONFIG_JSON = r'C:\projects\RC_column\model_config.json'
CONFIG_JSON = r''


def locate_json():
    if CONFIG_JSON.strip():
        candidate = os.path.abspath(os.path.normpath(CONFIG_JSON.strip().strip('"')))
        if not os.path.isfile(candidate):
            raise IOError('CONFIG_JSON does not exist: {}'.format(candidate))
        return candidate

    args = sys.argv
    if '--' in args:
        i = args.index('--')
        if i + 1 < len(args) and os.path.isfile(args[i + 1]):
            return os.path.abspath(args[i + 1])

    candidate = getInput(
        prompt='Enter full path to Grasshopper model_config.json (SI; X axis):',
        default=os.path.join(os.getcwd(), 'model_config.json'))
    if candidate is None or not str(candidate).strip():
        raise RuntimeError('Cancelled: no JSON path entered.')
    candidate = os.path.abspath(os.path.normpath(str(candidate).strip().strip('"')))
    if not os.path.isfile(candidate):
        raise IOError('Cannot find JSON file: {}'.format(candidate))
    return candidate


def finite_number(raw, name):
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError('{} must be numeric.'.format(name))
    # math.isfinite is not available in every Abaqus Python version.
    if math.isnan(value) or math.isinf(value):
        raise ValueError('{} must be finite.'.format(name))
    return value


def positive(raw, name):
    v = finite_number(raw, name)
    if v <= 0.0:
        raise ValueError('{} must be positive.'.format(name))
    return v


def poisson(raw, name):
    v = finite_number(raw, name)
    if not (-1.0 < v < 0.5):
        raise ValueError('{} must satisfy -1 < nu < 0.5.'.format(name))
    return v


def polygon_yz(raw):
    """Each (Y,Z) point is entered as (sketch X,sketch Y) before rotation."""
    pts = []
    for p in raw:
        if len(p) < 2:
            raise ValueError('Invalid cross-section vertex: {}'.format(p))
        pts.append((finite_number(p[0], 'section Y'),
                    finite_number(p[1], 'section Z')))
    if len(pts) > 2 and pts[0] == pts[-1]:
        pts.pop()
    if len(pts) < 3:
        raise ValueError('Concrete YZ cross-section must have at least 3 vertices.')
    if len(set(pts)) < 3:
        raise ValueError('Concrete YZ cross-section has too few distinct vertices.')
    signed_area2 = 0.0
    for i in range(len(pts)):
        j = (i + 1) % len(pts)
        signed_area2 += pts[i][0]*pts[j][1] - pts[j][0]*pts[i][1]
    if abs(signed_area2) < 1.0e-12:  # m^2 times 2
        raise ValueError('Concrete YZ cross-section has zero or negligible area.')
    if signed_area2 < 0:
        pts.reverse()
    return pts


def distance_squared(p, q):
    """Squared 3D distance without sum(generator), for Abaqus compatibility."""
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    dz = p[2] - q[2]
    return dx * dx + dy * dy + dz * dz


def rebar_segments(curves, closed, name, L):
    """Convert actual global XYZ curve vertices in m into Abaqus line segments."""
    segments = []
    n_curves = 0
    x_tolerance = 1.0e-6  # meters
    for curve in curves:
        n_curves += 1
        pts = []
        for p in curve:
            if len(p) < 3:
                raise ValueError('{} curve {} has a malformed XYZ point.'.format(name, n_curves))
            q = (finite_number(p[0], 'X'), finite_number(p[1], 'Y'),
                 finite_number(p[2], 'Z'))
            if not -x_tolerance <= q[0] <= L + x_tolerance:
                raise ValueError(
                    '{} curve {}: X={} m outside concrete X=[0,{}] m.'.format(
                        name, n_curves, q[0], L))
            if not pts or distance_squared(q, pts[-1]) > 1.0e-18:
                pts.append(q)
        if closed and len(pts) > 2 and distance_squared(pts[0], pts[-1]) <= 1.0e-18:
            pts.pop()
        if len(pts) < (3 if closed else 2):
            raise ValueError('{} curve {}: too few distinct points.'.format(name, n_curves))
        if closed:
            pts.append(pts[0])
        for i in range(len(pts) - 1):
            p0, p1 = pts[i], pts[i + 1]
            if distance_squared(p0, p1) > 1.0e-18:
                segments.append((p0, p1))
    return segments, n_curves


def wire_part(model, name, segments, section):
    if not segments:
        return None
    part = model.Part(name=name, dimensionality=THREE_D, type=DEFORMABLE_BODY)
    part.WirePolyLine(points=tuple(segments), mergeType=SEPARATE, meshable=ON)
    region = part.Set(name='ALL_REBAR_EDGES', edges=part.edges)
    part.SectionAssignment(region=region, sectionName=section)
    return part


def unique_name(base, existing):
    if base not in existing:
        return base
    i = 2
    while '{}_{}'.format(base, i) in existing:
        i += 1
    return '{}_{}'.format(base, i)


def unique_cae_file(folder, base):
    candidate = os.path.join(folder, base + '.cae')
    i = 2
    while os.path.exists(candidate):
        candidate = os.path.join(folder, '{}_{}.cae'.format(base, i))
        i += 1
    return candidate


def read_config(path):
    with open(path, 'r') as f:
        cfg = json.load(f)

    if cfg.get('schema_version') != 2:
        raise ValueError('Expected Grasshopper schema_version=2 (X axis and SI).')
    units = cfg.get('units')
    if units != {'length': 'm', 'force': 'N', 'stress': 'Pa'}:
        raise ValueError('Expected SI units {length:m, force:N, stress:Pa}. Got: {}'.format(units))
    geo = cfg['geometry']
    if geo.get('column_axis') != 'X':
        raise ValueError('Expected geometry.column_axis = X.')
    origin = geo.get('column_origin_xyz', [0.0, 0.0, 0.0])
    if len(origin) != 3:
        raise ValueError('column_origin_xyz must have exactly three values.')
    for val in origin:
        if abs(finite_number(val, 'origin')) > 1.0e-9:
            raise ValueError('Only column_origin_xyz=[0,0,0] is supported.')

    L = positive(geo['column_length'], 'ColumnLength [m]')
    yz = polygon_yz(geo['concrete_boundary_yz'])
    mat = cfg['materials']
    Ec = positive(mat['concrete']['E'], 'Concrete Ec [Pa]')
    Es = positive(mat['steel']['E'], 'Steel Es [Pa]')
    nu_c = poisson(mat['concrete']['nu'], 'NuC')
    nu_s = poisson(mat['steel']['nu'], 'NuS')
    sec = cfg['sections']
    dlong = positive(sec['longitudinal_diameter'], 'Long diameter [m]')
    dstir = positive(sec['stirrup_diameter'], 'Stirrup diameter [m]')
    if not (0.001 <= dlong <= 0.1 and 0.001 <= dstir <= 0.1):
        raise ValueError('Diameters must be in METERS (e.g. 0.016 / 0.008).')

    longitudinal, n_long = rebar_segments(
        geo.get('longitudinal_curves', []), False, 'Longitudinal', L)
    stirrups, n_stir = rebar_segments(
        geo.get('stirrup_curves', []), True, 'Stirrup', L)

    return (L, yz, Ec, Es, nu_c, nu_s, dlong, dstir,
            longitudinal, stirrups, n_long, n_stir)


def main():
    config_path = locate_json()
    (L, yz, Ec, Es, nu_c, nu_s, dlong, dstir,
     longitudinal, stirrups, n_long, n_stir) = read_config(config_path)

    model_name = unique_name('GH_RC_Column_X_SI', mdb.models)
    model = mdb.Model(name=model_name)

    conc_mat = model.Material(name='Concrete_Elastic')
    conc_mat.Elastic(table=((Ec, nu_c),))
    steel_mat = model.Material(name='Steel_Elastic')
    steel_mat.Elastic(table=((Es, nu_s),))

    model.HomogeneousSolidSection(name='ConcreteSection', material='Concrete_Elastic')
    model.TrussSection(name='LongitudinalSection', material='Steel_Elastic',
                       area=math.pi * dlong**2 / 4.0)
    model.TrussSection(name='StirrupSection', material='Steel_Elastic',
                       area=math.pi * dstir**2 / 4.0)

    # Abaqus sketches in local XY and extrudes along local Z.
    # Here sketch X is GLOBAL Y; sketch Y is GLOBAL Z after assembly rotation.
    ymin = yz[0][0]
    ymax = yz[0][0]
    zmin = yz[0][1]
    zmax = yz[0][1]
    for y, z in yz[1:]:
        ymin = min(ymin, y)
        ymax = max(ymax, y)
        zmin = min(zmin, z)
        zmax = max(zmax, z)
    size = max(ymax - ymin, zmax - zmin, 0.1)
    sketch = model.ConstrainedSketch(name='ConcreteSection_YZ', sheetSize=4.0*size)
    for i, pt in enumerate(yz):
        sketch.Line(point1=pt, point2=yz[(i + 1) % len(yz)])

    concrete = model.Part(name='Concrete', dimensionality=THREE_D,
                          type=DEFORMABLE_BODY)
    concrete.BaseSolidExtrude(sketch=sketch, depth=L)
    concrete_set = concrete.Set(name='ALL_CONCRETE', cells=concrete.cells)
    concrete.SectionAssignment(region=concrete_set, sectionName='ConcreteSection')

    longitudinal_part = wire_part(model, 'Longitudinal', longitudinal,
                                  'LongitudinalSection')
    stirrup_part = wire_part(model, 'Stirrups', stirrups, 'StirrupSection')

    assembly = model.rootAssembly
    assembly.DatumCsysByDefault(CARTESIAN)
    assembly.Instance(name='Concrete-1', part=concrete, dependent=ON)

    # Local (x,y,z) -> global (z,x,y), i.e. concrete length Z -> GLOBAL X.
    # Positive right-hand +120 degrees about the (1,1,1) diagonal.
    assembly.rotate(instanceList=('Concrete-1',),
                    axisPoint=(0.0, 0.0, 0.0),
                    axisDirection=(1.0, 1.0, 1.0), angle=120.0)

    # GH wires are already GLOBAL XYZ; do not transform them again.
    if longitudinal_part is not None:
        assembly.Instance(name='Longitudinal-1', part=longitudinal_part, dependent=ON)
    if stirrup_part is not None:
        assembly.Instance(name='Stirrups-1', part=stirrup_part, dependent=ON)
    assembly.regenerate()

    output_path = unique_cae_file(os.path.dirname(config_path),
                                  'GH_RC_Column_X_SI')
    mdb.saveAs(pathName=output_path)

    print('Abaqus model: {}'.format(model_name))
    print('Saved: {}'.format(output_path))
    print('Global concrete axis: X=0 to X={:.6g} m'.format(L))
    print('Concrete sketch coordinates = (global Y, global Z); instance rotated +120 deg')
    print('Longitudinal curves: {}; wire edges: {}'.format(n_long, len(longitudinal)))
    print('Stirrup curves: {}; wire edges: {}'.format(n_stir, len(stirrups)))
    print('Diameters: long = {:.6g} m, stirrup = {:.6g} m'.format(dlong, dstir))
    print('Materials: Ec={:.6g} Pa, Es={:.6g} Pa (Poisson: {}, {})'.format(
        Ec, Es, nu_c, nu_s))
    print('NOT created: mesh sizes/seeds, mesh, element types, embedded constraints,')
    print('             loads, supports/BCs, steps, jobs, or analysis outputs.')
    print('Inspect ASSEMBLY: concrete and rebar are in the same global XYZ system.')


if __name__ == '__main__':
    main()
