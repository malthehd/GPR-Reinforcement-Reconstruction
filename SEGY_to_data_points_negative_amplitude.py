#! python 3
# -*- coding: utf-8 -*-
# venv: gpr-rebar-432
# r: scipy
#
# Rhino 8 Python 3 / Grasshopper: retain input and output names in OUTPUT_NAMES.
# Configure scan directions and constants below; do not change without checking
# against the Proceq acquisition geometry.

"""Read Proceq SEG-Y scans and output individual GPR reinforcement picks.

Rhino 8 / Grasshopper Python 3. Reconstruction and +/-20 mm grouping happen
in Grasshopper, NOT here. The script fits negative-amplitude hyperbolic reflections, estimates
one common time-zero from eight longitudinal scans, transforms picks to XYZ,
and calculates mean top/side stirrup cover and a 432 x 432 mm reference section.

Transverse scans also support constrained one-sided fits for corner bars.
These fits are geometry-guided candidates, not independent confirmations.
Depth estimates depend on epsilon_r, t0 and the assumed bar-radius correction.
Only negative signed-amplitude reflections are used for candidate detection
and hyperbola fitting (not the polarity-independent Hilbert envelope).

GH inputs: folder, run, makePipes (legacy ignored), longDiameter (=16 if set).
GH outputs: see OUTPUT_NAMES; legacy empty outputs remain for compatibility.
All dimensions are in mm; time values are in ns.
"""
import math
import re
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates
from scipy.signal import find_peaks
SPEED_MM_NS = 299.792458
DATA_SUBFOLDER = 'GPR_Data'
COLUMN_LENGTH_MM = 2950.0
COLUMN_WIDTH_MM = COLUMN_HEIGHT_MM = 480.0
DIELECTRIC_EPSILON = 7.8
STIRRUP_DIAMETER_MM = 8.0
STIRRUP_CENTRE_WIDTH_MM = STIRRUP_CENTRE_HEIGHT_MM = 432.0
LONGITUDINAL_DIAMETER_MM = 16.0
CROSS_SCAN_FIRST_X_MM = 100.0
MIN_TRANSVERSE_MM = 300.0
MAX_TRANSVERSE_MM = 800.0
FILE_ASSIGNMENTS = {'CODB8E~1.SGY': ('side', 2), 'CO72DA~1.SGY': ('side', 3), 'CO7E82~1.SGY': ('side', 4), 'CODFC6~1.SGY': ('side', 5)}
TOP_TRANSVERSE_REVERSE = False
SIDE_TRANSVERSE_REVERSE = False
TOP_TRANSVERSE_ALTERNATING = False
SIDE_TRANSVERSE_ALTERNATING = False
TOP_TRANSVERSE_REVERSE_LINES = ()
SIDE_TRANSVERSE_REVERSE_LINES = ()
TOP_LONGITUDINAL_REVERSE = False
SIDE_LONGITUDINAL_REVERSE = False
TOP_TRANSVERSE_START_MM = 0.0
SIDE_TRANSVERSE_START_MM = 0.0
REQUIRE_COMPLETE_TRANSVERSE_MAPPING = False
LONG_APEX_RAW_NS = (1.25, 1.85)
CROSS_APEX_RAW_NS = (1.2, 2.1)
CROSS_MIN_PEAK_SPACING_MM = 50.0
CROSS_MIN_APEX_SCORE = 1.5
CROSS_MIN_APEX_PROMINENCE = 1.0
CROSS_MIN_FIT_SCORE = 0.6
ENABLE_HALF_HYPERBOLA_FIT = True
HALF_EDGE_APEX_TOL_MM = 45.0
HALF_APEX_STEP_MM = 5.0
HALF_MAX_APEX_OUTSIDE_SCAN_MM = 70.0
HALF_REFLECTOR_DEPTH_TOL_MM = 28.0
HALF_REFLECTOR_DEPTH_STEP_MM = 2.0
HALF_BRANCH_MIN_OFFSET_MM = 10.0
HALF_BRANCH_MAX_OFFSET_MM = 105.0
HALF_BRANCH_SAMPLE_STEP_MM = 5.0
HALF_MIN_VALID_SAMPLES = 11
HALF_MIN_CONTRAST_QUALITY = 1.6
HALF_MIN_LOCAL_RIDGE_SCORE = 0.75
HALF_MIN_COMBINED_SCORE = 1.1
HALF_FULL_APEX_ACCEPT_TOL_MM = 12.0
HALF_REPLACE_NEARBY_FULL_MM = 36.0
MIN_LONG_SCAN_MM = 2500.0
DEFAULT_CONFIG = {
    'epsilon_r': DIELECTRIC_EPSILON, 'bar_diameter_mm': STIRRUP_DIAMETER_MM,
    'apex_window_raw_ns': LONG_APEX_RAW_NS,
    'min_peak_spacing_mm': 80., 'min_apex_score': 1.7,
    'min_apex_prominence': 1.3,
    'expected_longitudinal_scans': 8, 'peak_mode':'first_negative',
    'first_peak_window_ns':[0.45,1.17],
    'first_peak_min_relative_amplitude':.23,
    'first_peak_min_relative_prominence':.10,
    'time_zero_max_peak_spread_ns':.05,
    'column_length_mm':COLUMN_LENGTH_MM, 'column_width_mm':COLUMN_WIDTH_MM,
    'column_height_mm':COLUMN_HEIGHT_MM,
}

def read_vendor_csv(path):
    raw = Path(path).read_bytes()
    if raw[:2] in (b'\xff\xfe', b'\xfe\xff'):
        txt = raw.decode('utf-16')
    else:
        try:
            txt = raw.decode('utf-8-sig')
        except UnicodeDecodeError:
            txt = raw.decode('utf-16-le')
    settings, line_distances = ({}, {})
    in_distances = False
    for row in txt.splitlines():
        parts = row.strip().split('\t')
        if not parts:
            continue
        key = parts[0].strip()
        if key.upper() == 'SCAN DISTANCE [M]':
            in_distances = True
            continue
        if len(parts) < 2:
            continue
        value = parts[1].strip()
        if in_distances and re.fullmatch('Line\\s+\\d+', key, re.I):
            line_distances[int(re.search('\\d+', key).group())] = float(value)
        else:
            settings[key] = value

    def value(name):
        if name not in settings:
            raise ValueError('CSV setting missing: {} ({})'.format(name, path))
        return float(settings[name])
    return {
        'samples': int(value('Scan Length [samples/scan]')),
        'dt_ns': 1.0 / value('Sampling Rate [GHz]'),
        'dx_mm': 10.0 / value('Repetition Rate [scans/cm]'),
        'line_spacing_mm': 10.0 * value('Grid Spacing - X [cm]'),
        'antenna_sep_mm': 10.0 * value('Antenna Spacing [cm]'),
        'lengths_m': line_distances,
    }

def read_segy_ieee_float32(path):
    """Read fixed-trace SEG-Y format=5 (IEEE32), as verified for uploaded SGYs."""
    blob = Path(path).read_bytes()
    if len(blob) < 3600:
        raise ValueError('SEG-Y too short: {}'.format(path))
    hdr = blob[3200:3600]
    ns = int.from_bytes(hdr[20:22], 'big')
    fmt = int.from_bytes(hdr[24:26], 'big')
    ext = int.from_bytes(hdr[304:306], 'big', signed=True)
    if fmt != 5 or ext != 0 or ns < 100:
        raise ValueError('Unsupported SEG-Y: format={}, samples={}, extra headers={} ({})'.format(fmt, ns, ext, path))
    frame_bytes = 240 + 4 * ns
    length = len(blob) - 3600
    if length % frame_bytes:
        raise ValueError('Invalid or variable trace length in {}'.format(path))
    ntr = length // frame_bytes
    raw = np.frombuffer(blob, dtype=np.uint8, offset=3600).reshape(ntr, frame_bytes)
    return np.ascontiguousarray(raw[:, 240:]).view('>f4').reshape(ntr, ns).T.astype(np.float32)

def tt_curve(offset_mm, depth_mm, v_mm_ns, sep_mm, t0_ns):
    """Bistatic (separate transmitting/receiving centres) curved reflection."""
    return t0_ns + np.hypot(offset_mm - sep_mm / 2.0, depth_mm) / v_mm_ns + np.hypot(offset_mm + sep_mm / 2.0, depth_mm) / v_mm_ns

# Correct the bistatic Tx/Rx geometry (36 mm from Proceq metadata).

def depth_from_apex(raw_apex_ns, v_mm_ns, sep_mm, t0_ns):
    halfpath = v_mm_ns * (raw_apex_ns - t0_ns) / 2.0
    if halfpath < sep_mm / 2.0:
        return float('nan')
    return math.sqrt(max(0.0, halfpath ** 2 - (sep_mm / 2.0) ** 2))


# --- SIGNAL INTERPRETATION ---

def normalized_radar_response(matrix):
    """Contrast of NEGATIVE signed-amplitude troughs only.

    A Gaussian filter reduces noise. Positive amplitudes are set to zero,
    so neither apex detection nor hyperbola fitting can be driven by a
    positive pulse. Subtract the per-time-sample median background, then
    normalize by robust raw-signal variation across traces. The signed
    variation also provides a non-zero noise scale when few traces contain
    negative reflections. This replaces the previous Hilbert-envelope method.
    """
    a = gaussian_filter(matrix.astype(np.float64), (0.5, 0.6))
    negative = np.maximum(-a, 0.0)
    median_negative = np.median(negative, axis=1, keepdims=True)
    median_raw = np.median(a, axis=1, keepdims=True)
    mad = 1.4826 * np.median(np.abs(a - median_raw), axis=1, keepdims=True)
    floor = max(1e-12, float(np.median(mad)) * 0.12)
    contrast = (negative - median_negative) / np.maximum(mad, floor)
    return gaussian_filter(np.clip(contrast, -2.0, 15.0), (1.2, 1.2))

def detect_hyperbolas(matrix, meta, cfg):
    """Near-surface apex detection + physical hyperbola template scoring.
    For each signal class, cfg["bar_diameter_mm"] is the correct known bar Ø.
    """
    contrast = normalized_radar_response(matrix)
    nt, nx = contrast.shape
    dt, dx, sep = (meta['dt_ns'], meta['dx_mm'], meta['antenna_sep_mm'])
    lo, hi = map(float, cfg['apex_window_raw_ns'])
    valid = np.flatnonzero((np.arange(nt) * dt >= lo) & (np.arange(nt) * dt <= hi))
    if len(valid) < 4:
        raise ValueError('No GPR samples within apex window')
    response = gaussian_filter1d(np.max(contrast[valid, :], axis=0), 2)
    peaks,_ = find_peaks(
        response,
        distance=max(2,round(float(cfg.get('min_peak_spacing_mm',80.))/dx)),
        height=float(cfg.get('min_apex_score',1.7)),
        prominence=float(cfg.get('min_apex_prominence',1.3)),
    )
    vel = SPEED_MM_NS / math.sqrt(float(cfg['epsilon_r']))
    t0 = float(cfg['time_zero_ns'])
    offsets = np.arange(-80.0, 80.01, 5.0)
    picks = []
    for peak in peaks:
        v = contrast[:, max(0, peak - 2):min(nx, peak + 3)].mean(axis=1)
        seed = valid[np.argmax(v[valid])] * dt
        trial_times = np.arange(max(lo, seed - 0.11), min(hi, seed + 0.11) + dt / 2, dt / 2)
        best = None
        for t_apex in trial_times:
            depth = depth_from_apex(t_apex, vel, sep, t0)
            if not np.isfinite(depth) or depth <= 0:
                continue
            t_curve = tt_curve(offsets, depth, vel, sep, t0)
            coords = np.vstack((t_curve / dt, peak + offsets / dx))
            good = (coords[1] >= 0) & (coords[1] < nx) & (coords[0] >= 0) & (coords[0] < nt)
            if np.count_nonzero(good) < 12:
                continue
            vals = map_coordinates(contrast, coords[:, good], order=1, mode='nearest')
            weights = np.exp(-0.5 * (offsets[good] / 61.0) ** 2)
            quality = float(np.average(vals, weights=weights))
            score = quality - 0.55 * abs(t_apex - seed) / 0.11
            if best is None or score > best[0]:
                best = (score, float(t_apex), float(depth))
        if best is None:
            continue
        _, t_apex, d_reflector = best
        picks.append({'x_mm': float(peak*dx),
                      'depth_reflection_mm':d_reflector,
                      'depth_bar_center_mm':d_reflector+float(cfg['bar_diameter_mm'])/2.0,
                      't_apex_raw_ns':t_apex,
                      'score':float(response[peak]), 'fit_score':float(best[0]),
                       'fit_mode':'full_hyperbola'})
    return picks

# One-branch fits use the expected corner geometry as a PRIOR, not proof.
# Their fit_mode must remain visible for engineering review.

def detect_one_sided_edge_hyperbolas(matrix, meta, cfg, existing_picks, expected_apexes_mm, expected_center_depth_mm):
    """Constrained one-branch fitting for the extreme longitudinal bars ONLY.

    This does NOT assert that an edge bar exists. A fitted template must show an
    amplitude ridge along the visible branch and be stronger than neighbouring
    time-offset curves. The candidate apex and reflector depth are *constrained*
    by the idealized stirrup-to-Ø16 contact geometry and should NOT be presented
    as an independent measurement. A flag is attached for QA.

    expected_apexes_mm: physical transverse scan coordinates of near/far corner
    bar centers, after scan-direction correction, with zero at matrix column 0.
    expected_center_depth_mm: the straight bar's assumed GPR-facing centre depth.
    """
    if matrix.shape[1] < 15:
        return []
    contrast = normalized_radar_response(matrix)
    nt, nx = contrast.shape
    dt, dx, sep = (meta['dt_ns'], meta['dx_mm'], meta['antenna_sep_mm'])
    v = SPEED_MM_NS / math.sqrt(float(cfg['epsilon_r']))
    t0 = float(cfg['time_zero_ns'])
    diameter = float(cfg['bar_diameter_mm'])
    nominal_depth = float(expected_center_depth_mm) - diameter / 2.0
    low_depth = max(3.0, nominal_depth - HALF_REFLECTOR_DEPTH_TOL_MM)
    high_depth = min(90.0, nominal_depth + HALF_REFLECTOR_DEPTH_TOL_MM)
    if high_depth <= low_depth:
        return []
    depth_candidates = np.arange(low_depth, high_depth + 1e-06, HALF_REFLECTOR_DEPTH_STEP_MM)
    extent_mm = (nx - 1) * dx
    results = []
    offsets = np.arange(HALF_BRANCH_MIN_OFFSET_MM, HALF_BRANCH_MAX_OFFSET_MM + 0.01, HALF_BRANCH_SAMPLE_STEP_MM)
    for edge, expected_apex in zip(('left', 'right'), expected_apexes_mm):
        expected_apex = float(expected_apex)
        known = [p for p in existing_picks if p.get('fit_score', -1000000000.0) >= CROSS_MIN_FIT_SCORE and abs(float(p['x_mm']) - expected_apex) <= HALF_FULL_APEX_ACCEPT_TOL_MM]
        if known:
            continue
        lo_x = max(-HALF_MAX_APEX_OUTSIDE_SCAN_MM, expected_apex - HALF_EDGE_APEX_TOL_MM)
        hi_x = min(extent_mm + HALF_MAX_APEX_OUTSIDE_SCAN_MM, expected_apex + HALF_EDGE_APEX_TOL_MM)
        x_candidates = np.arange(lo_x, hi_x + 0.01, HALF_APEX_STEP_MM)
        if not len(x_candidates):
            continue
        best = None
        signs = 1.0 if edge == 'left' else -1.0
        signed_offsets = signs * offsets
        for x0 in x_candidates:
            locs = x0 + signed_offsets
            good_x = (locs >= 0.0) & (locs <= extent_mm)
            if np.count_nonzero(good_x) < HALF_MIN_VALID_SAMPLES:
                continue
            xcol = locs[good_x] / dx
            for depth in depth_candidates:
                times = tt_curve(signed_offsets[good_x], depth, v, sep, t0)
                t_apex = float(tt_curve(0.0, depth, v, sep, t0))
                if not cfg['apex_window_raw_ns'][0] <= t_apex <= cfg['apex_window_raw_ns'][1]:
                    continue
                tidx = times / dt
                valid = (tidx > 5.0) & (tidx < nt - 6.0)
                if np.count_nonzero(valid) < HALF_MIN_VALID_SAMPLES:
                    continue
                xs = xcol[valid]
                ts = tidx[valid]
                response = map_coordinates(contrast, np.vstack((ts, xs)), order=1, mode='nearest')
                weights = np.exp(-0.5 * (offsets[good_x][valid] / 78.0) ** 2)
                quality = float(np.average(response, weights=weights))
                dt_shift = 0.15 / dt
                above = map_coordinates(contrast, np.vstack((ts - dt_shift, xs)), order=1, mode='nearest')
                below = map_coordinates(contrast, np.vstack((ts + dt_shift, xs)), order=1, mode='nearest')
                ridge = quality - max(float(np.average(above, weights=weights)), float(np.average(below, weights=weights)))
                x_penalty = 0.35 * abs(x0 - expected_apex) / max(1.0, HALF_EDGE_APEX_TOL_MM)
                z_penalty = 0.25 * abs(depth - nominal_depth) / max(1.0, HALF_REFLECTOR_DEPTH_TOL_MM)
                combined = quality + 0.6 * ridge - x_penalty - z_penalty
                if best is None or combined > best['combined']:
                    best = {'combined': float(combined), 'x': float(x0),
                            'depth': float(depth), 't_apex': t_apex,
                            'quality': quality, 'ridge': ridge,
                            'samples': int(np.count_nonzero(valid))}
        if best is None:
            continue
        if best['quality'] < HALF_MIN_CONTRAST_QUALITY or best['ridge'] < HALF_MIN_LOCAL_RIDGE_SCORE or best['combined'] < HALF_MIN_COMBINED_SCORE:
            continue
        results.append({
            'x_mm': best['x'], 'depth_reflection_mm': best['depth'],
            'depth_bar_center_mm': best['depth'] + diameter/2.0,
            't_apex_raw_ns': best['t_apex'],
            'score': best['quality'], 'fit_score': best['combined'],
            'fit_mode': 'one_sided_' + edge,
            'one_sided_visible_branch': 'right' if edge == 'left' else 'left',
            'one_sided_quality': best['quality'],
            'one_sided_ridge_score': best['ridge'],
            'one_sided_samples': best['samples'],
            'expected_edge_apex_mm': expected_apex,
            'one_sided_assumption': 'expected_corner_position_and_depth',
        })
    return results


# --- TIME-ZERO CALIBRATION ---

def first_peak_time_ns(scan, dt_ns, cfg):
    """First SIGNIFICANT NEGATIVE amplitude trough of the early radar pulse.

    Use a median A-scan per SGY. Find the first sufficiently prominent negative
    local extremum within the early-pulse window, then refine with a quadratic
    fit. A noise dip below the defined thresholds is not accepted as t0.
    The result is an operational reference, not absolute instrument delay.
    """
    ntr = scan.shape[1]
    a = max(0, int(ntr * 0.05))
    b = min(ntr, max(a + 1, int(ntr * 0.95)))
    median_trace = np.median(scan[:, a:b], axis=1).astype(float)
    time_ns = np.arange(len(median_trace), dtype=float) * float(dt_ns)
    lo, hi = map(float, cfg.get('first_peak_window_ns', [0.45, 1.17]))
    valid = np.flatnonzero((time_ns >= lo) & (time_ns <= hi))
    if len(valid) < 5:
        raise ValueError('First-peak window [%.4f, %.4f] ns contains fewer than 5 samples' % (lo, hi))
    max_amp = float(np.max(np.abs(median_trace[valid])))
    if not np.isfinite(max_amp) or max_amp <= 1e-12:
        raise ValueError('No measurable early radar pulse in the specified time window')
    mode = str(cfg.get('peak_mode', 'first_negative')).lower().strip()
    if mode in ('first_positive', 'first_negative'):
        threshold = max_amp * float(cfg.get('first_peak_min_relative_amplitude', 0.23))
        prominence = max_amp * float(cfg.get('first_peak_min_relative_prominence', 0.1))
        polarity = -1.0 if mode == 'first_negative' else 1.0
        fit_signal = polarity * median_trace
        candidates, _ = find_peaks(fit_signal, prominence=prominence)
        candidates = [int(i) for i in candidates if valid[0] <= i <= valid[-1] and fit_signal[i] >= threshold]
        if not candidates:
            raise ValueError('No SIGNIFICANT %s early peak in %.2f..%.2f ns. Inspect waveform/thresholds.' % (mode, lo, hi))
        k = candidates[0]
    elif mode == 'dominant_absolute':
        k = int(valid[np.argmax(np.abs(median_trace[valid]))])
        fit_signal = np.abs(median_trace)
    else:
        raise ValueError('Unknown peak_mode=%r; choose first_negative, first_positive or dominant_absolute' % mode)
    dk = 0.0
    if 0 < k < len(median_trace) - 1:
        a0, a1, a2 = map(float, (fit_signal[k - 1], fit_signal[k], fit_signal[k + 1]))
        denominator = a0 - 2 * a1 + a2
        if abs(denominator) > 1e-15:
            dk = float(np.clip(0.5 * (a0 - a2) / denominator, -0.5, 0.5))
    return {
        'peak_ns': float((k+dk)*dt_ns),
        'sample_peak_ns':float(k*dt_ns),
        'peak_index':int(k),
        'amplitude':float(median_trace[k]),
        'relative_amplitude':float(abs(median_trace[k])/max_amp),
        'peak_mode':mode
    }

def calculate_common_time_zero(scan_records, cfg):
    """Calculate one peak per longitudinal SGY and arithmetic mean of 8 peaks."""
    expected = int(cfg.get('expected_longitudinal_scans', 8))
    if len(scan_records) != expected:
        raise ValueError('Expected %d LONGITUDINAL SGYs for t0, got %d; abort to avoid averaging the wrong scans.' % (expected, len(scan_records)))
    peaks = []
    for rec in scan_records:
        r = first_peak_time_ns(rec['matrix'], rec['meta']['dt_ns'], cfg)
        r.update({'file': rec['file'].name, 'area': rec['area'], 'line': rec['line']})
        peaks.append(r)
    vals = np.asarray([r['peak_ns'] for r in peaks], dtype=float)
    average = float(vals.mean())
    std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    range_ns = float(vals.max() - vals.min())
    warnings = []
    if range_ns > float(cfg.get('time_zero_max_peak_spread_ns', 0.05)):
        warnings.append('WARNING: peak times differ by %.4fns (> tolerance); verify same pulse in every scan.' % range_ns)
    return (average, peaks, std, range_ns, warnings)


# --- INPUT DISCOVERY / PROCEQ METADATA ---

def face_from_folder_name(name):
    """Recognize Danish or English top/side folder names (case-insensitive)."""
    name = re.sub('[ _\\-]+', '', str(name).lower())
    if 'top' in name or 'overside' in name or name == 'overflade':
        return 'top'
    if 'side' in name or 'siden' in name:
        return 'side'
    return None

def face_scan_dirs(root):
    """Use face subdirectories when present. No SGY filename needs 'top'/'side'."""
    dirs = {}
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        face = face_from_folder_name(path.name)
        if face:
            if face in dirs:
                raise ValueError('Multiple folders interpreted as %s: %s and %s' % (face, dirs[face].name, path.name))
            dirs[face] = path
    return dirs

def resolve_data_folder(requested=None):
    """Accept saved .gh directory, GPR_Data, or either face subfolder."""
    bases = []
    if requested and str(requested).strip():
        bases = [Path(str(requested).strip().strip('"')).expanduser()]
    else:
        try:
            import Grasshopper
            doc = Grasshopper.Instances.ActiveCanvas.Document
            saved = getattr(doc, 'FilePath', None)
            if saved:
                bases.append(Path(str(saved)).parent)
        except Exception:
            pass
        if '__file__' in globals():
            bases.append(Path(__file__).resolve().parent)
    candidates = []
    for base in bases:
        if base.is_file():
            base = base.parent
        candidates.extend([base / DATA_SUBFOLDER, base])
        if face_from_folder_name(base.name):
            candidates.append(base.parent)
    for root in candidates:
        if not root.is_dir():
            continue
        csvs = list(root.glob('*.csv')) + list(root.glob('*.CSV'))
        face_dirs = face_scan_dirs(root)
        nested_csvs = [c for d in face_dirs.values() for c in d.iterdir() if c.is_file() and c.suffix.lower() == '.csv']
        if len(csvs) + len(nested_csvs) >= 2 and (face_dirs or any((f.suffix.lower() in ('.sgy', '.segy') for f in root.iterdir() if f.is_file()))):
            return root.resolve()
    raise FileNotFoundError(
        'GPR_Data not found with 2 CSVs and SGY files. Expected GPR_Data/Topscanning/ '
        'and GPR_Data/Sidescanning/ next to the saved .gh file. '
        'Or set folder to project directory or GPR_Data. Checked: %s' % ', '.join(str(p) for p in candidates))

def metadata_face_from_content(path):
    """Detect old manufacturer CSV name fields without assuming the filename."""
    raw = path.read_bytes()
    if raw[:2] in (b'\xff\xfe', b'\xfe\xff'):
        txt = raw.decode('utf-16')
    else:
        try:
            txt = raw.decode('utf-8-sig')
        except UnicodeDecodeError:
            txt = raw.decode('utf-16-le')
    first = [line for line in txt.splitlines() if line.lower().startswith('file name\t')]
    info = ' '.join([path.stem] + first).lower()
    if 'top' in info:
        return 'top'
    if 'side' in info or 'column' in info or path.stem.lower().startswith('co'):
        return 'side'
    return None

def load_face_metadata(root):
    dirs = face_scan_dirs(root)
    files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() == '.csv']
    for d in dirs.values():
        files.extend((p for p in d.iterdir() if p.is_file() and p.suffix.lower() == '.csv'))
    files = sorted(files)
    if len(files) != 2:
        raise ValueError('Expected exactly TWO original Proceq CSV metadata files in GPR_Data or its face folders, found %d: %s.' %
                         (len(files),', '.join(str(p.relative_to(root)) for p in files)))
    result = {}
    paths = {}
    for p in files:
        assigned = next((face for face, d in dirs.items() if p.parent == d), None)
        from_name = metadata_face_from_content(p)
        if assigned and from_name and (assigned != from_name):
            raise ValueError('CSV face conflict: %s is under %s but CSV appears to be %s.' % (p.name, assigned, from_name))
        face = assigned or from_name
        if face is None:
            raise ValueError('Cannot classify CSV as TOP or SIDE: ' + str(p))
        if face in result:
            raise ValueError('Two CSVs classified as %s. Move one to the correct face folder or verify metadata.' % face)
        result[face] = read_vendor_csv(p)
        paths[face] = str(p.relative_to(root))
    if set(result) != {'top', 'side'}:
        raise ValueError('Both TOP and SIDE Proceq CSV metadata required.')
    return (result, paths)

def filename_scan_identity(path, parent_face=None):
    name = path.name
    manual = next((val for key, val in FILE_ASSIGNMENTS.items() if key.lower() == name.lower()), None)
    if manual:
        if parent_face and parent_face != manual[0]:
            raise ValueError('%s in %s folder but manual mapping says %s' % (name, parent_face, manual[0]))
        return manual
    match = re.search('(?i)(?:^|[_.\\s-])L(?:INE)?0*(\\d{1,3})(?=$|[_.\\s-])', path.stem)
    if match is None:
        match = re.search('(?i)(?:^|[_\\s-])LINE\\s*0*(\\d{1,3})\\b', path.stem)
    if match is None:
        return None
    line = int(match.group(1))
    label = path.name.lower()
    if 'top' in label:
        file_face = 'top'
    elif 'side' in label or 'column' in label or label.startswith('co'):
        file_face = 'side'
    else:
        file_face = None
    if parent_face and file_face and (parent_face != file_face):
        raise ValueError('Folder/filename face conflict: %s in %s folder' % (path.name, parent_face))
    return (parent_face or file_face, line) if parent_face or file_face else None

def gather_scans(root, vendor):
    dirs = face_scan_dirs(root)
    candidates = []
    if dirs:
        if set(dirs) != {'top', 'side'}:
            raise ValueError('Found only %s scan folder(s); both Topscanning and Sidescanning are required.' % ', '.join(dirs))
        for face, folder in dirs.items():
            candidates.extend(((p, face) for p in folder.iterdir() if p.is_file() and p.suffix.lower() in ('.sgy', '.segy')))
        stray = [p.name for p in root.iterdir() if p.is_file() and p.suffix.lower() in ('.sgy', '.segy')]
        if stray:
            raise ValueError('SGY files also in GPR_Data root; move into Topscanning/Sidescanning: ' + ', '.join(stray))
    else:
        candidates = [(p, None) for p in root.iterdir() if p.is_file() and p.suffix.lower() in ('.sgy', '.segy')]
    candidates.sort(key=lambda item: (str(item[0]).lower(), item[1] or ''))
    if not candidates:
        raise FileNotFoundError('No .sgy/.SGY in GPR_Data top/side folders: ' + str(root))
    records = []
    warnings = []
    unmapped = []
    unused = []
    claimed = {}
    for file, parent_face in candidates:
        identity = filename_scan_identity(file, parent_face=parent_face)
        if not identity:
            unmapped.append(str(file.relative_to(root)))
            continue
        face, line = identity
        label = str(file.relative_to(root))
        if (face, line) in claimed:
            raise ValueError('Two SGYs assigned to %s L%d: %s and %s' % (face, line, claimed[face, line], label))
        claimed[face, line] = label
        meta = vendor[face]
        if line == 1 or float(meta['lengths_m'].get(line, 0)) < 0.01:
            unused.append(label)
            continue
        if line not in meta['lengths_m']:
            warnings.append('File %s maps to %s L%d but L%d absent from CSV' % (label, face, line, line))
            continue
        expected_mm = 1000 * meta['lengths_m'][line]
        if expected_mm < MIN_TRANSVERSE_MM:
            warnings.append('Skipped line too short per CSV: ' + label)
            continue
        matrix = read_segy_ieee_float32(file)
        if matrix.shape[0] != meta['samples']:
            raise ValueError('SGY sample count mismatch for ' + label)
        scan_mm = (matrix.shape[1] - 1) * meta['dx_mm']
        if abs(scan_mm - expected_mm) > 30:
            warnings.append('%s: SGY length %.0f mm differs from metadata %.0f mm' % (label, scan_mm, expected_mm))
        orient = 'long' if scan_mm >= MIN_LONG_SCAN_MM else 'cross' if MIN_TRANSVERSE_MM <= scan_mm <= MAX_TRANSVERSE_MM else None
        if orient is None:
            warnings.append('Unclassified length %.0fmm: %s' % (scan_mm, label))
            continue
        if orient == 'long' and line not in (2, 3, 4, 5) or (orient == 'cross' and (not 8 <= line <= 36)):
            raise ValueError('%s L%d length %.0fmm: expected longitudinal L2-5 and transverse L8-36' % (face, line, scan_mm))
        records.append({'file': file, 'face': face, 'area': face, 'line': line, 'type': orient, 'matrix': matrix, 'meta': meta, 'scan_length_mm': scan_mm})
    warnings += ['UNMAPPED SGY: ' + name + ' (need ..._L008...SGY or FILE_ASSIGNMENTS)' for name in unmapped]
    if unmapped and REQUIRE_COMPLETE_TRANSVERSE_MAPPING:
        raise ValueError('SGY line identification missing for: ' + ', '.join(unmapped))
    records.sort(key=lambda r: (r['type'] != 'long', r['face'], r['line']))
    return (records, warnings, unmapped)


# --- DETECTION, COVER AND COORDINATE TRANSFORMATION ---
# TOP longitudinal: X=x, Y=line, Z=depth;
# SIDE longitudinal: X=L-x, Y=depth, Z=line;
# transverse scans use the same global X orientation convention.

def process_sgy_folder(folder_path=None, long_diameter_mm=None):
    """No geometric grouping. Return each fitted radar reflection separately."""
    root = resolve_data_folder(folder_path)
    cfg = dict(DEFAULT_CONFIG)
    dlong = LONGITUDINAL_DIAMETER_MM
    if long_diameter_mm is not None and abs(float(long_diameter_mm) - dlong) > 1e-06:
        raise ValueError('Longitudinal bar diameter is fixed at Ø16 mm; leave longDiameter empty or set 16.')
    vendor, metadata_names = load_face_metadata(root)
    recs, warnings, unmapped = gather_scans(root, vendor)
    longitudinal = [r for r in recs if r['type'] == 'long']
    transverse = [r for r in recs if r['type'] == 'cross']
    required = {(face, line) for face in ('top', 'side') for line in (2, 3, 4, 5)}
    available = {(r['face'], r['line']) for r in longitudinal}
    if available != required:
        raise ValueError('Missing longitudinal scans for common t0 and cover: %s. Check FILE_ASSIGNMENTS.' % sorted(required - available))
    timezero, peaks, _stdev, _spread, peakwarn = calculate_common_time_zero([dict(r, area=r['face']) for r in longitudinal], cfg)
    cfg['time_zero_ns'] = timezero
    warnings.extend(peakwarn)
    long_picks = []
    counts_long = []
    for rec in longitudinal:
        mat = rec['matrix']
        if rec['face'] == 'top' and TOP_LONGITUDINAL_REVERSE or (rec['face'] == 'side' and SIDE_LONGITUDINAL_REVERSE):
            mat = mat[:, ::-1]
        hits = detect_hyperbolas(mat, rec['meta'], cfg)
        face, line = (rec['face'], rec['line'])
        line_position = (line - 1) * rec['meta']['line_spacing_mm']
        for q in hits:
            q = dict(q)
            if face == 'top':
                X, Y, Z = (q['x_mm'], line_position, q['depth_bar_center_mm'])
            else:
                X, Y, Z = (COLUMN_LENGTH_MM - q['x_mm'], q['depth_bar_center_mm'], line_position)
            long_picks.append(dict(q, file=rec['file'].name, area=face, face=face, line=line, X_mm=float(X), Y_mm=float(Y), Z_mm=float(Z)))
        counts_long.append('%s L%02d: %d individual stirrup reflections' % (face, line, len(hits)))
    top_long = [p for p in long_picks if p['face'] == 'top']
    side_long = [p for p in long_picks if p['face'] == 'side']
    if not top_long or not side_long:
        raise ValueError('Cover calculation requires detected stirrup reflections on BOTH TOP and SIDE.')
    half_tie = STIRRUP_DIAMETER_MM / 2.0
    top_covers = np.asarray([p['depth_bar_center_mm'] - half_tie for p in top_long], float)
    side_covers = np.asarray([p['depth_bar_center_mm'] - half_tie for p in side_long], float)
    if not (np.isfinite(top_covers).all() and np.isfinite(side_covers).all()):
        raise ValueError('Non-finite cover estimate among original GPR picks.')
    mean_top_cover = float(np.mean(top_covers))
    mean_side_cover = float(np.mean(side_covers))
    common_Z_top = mean_top_cover + half_tie
    common_Y_near = mean_side_cover + half_tie
    common_Z_bottom = common_Z_top + STIRRUP_CENTRE_HEIGHT_MM
    common_Y_far = common_Y_near + STIRRUP_CENTRE_WIDTH_MM
    bottom_cover = COLUMN_HEIGHT_MM - common_Z_bottom - half_tie
    far_cover = COLUMN_WIDTH_MM - common_Y_far - half_tie
    common_section = dict(Y_near_mm=common_Y_near, Y_far_mm=common_Y_far,
                          Z_top_mm=common_Z_top, Z_bottom_mm=common_Z_bottom,
                          width_mm=STIRRUP_CENTRE_WIDTH_MM,
                          height_mm=STIRRUP_CENTRE_HEIGHT_MM)
    cover_statistics = {
        'top': {'n':len(top_covers), 'mean_mm':mean_top_cover,
                'min_mm':float(np.min(top_covers)), 'max_mm':float(np.max(top_covers)),
                'std_mm':float(np.std(top_covers,ddof=1)) if len(top_covers)>1 else 0.},
        'side': {'n':len(side_covers), 'mean_mm':mean_side_cover,
                 'min_mm':float(np.min(side_covers)), 'max_mm':float(np.max(side_covers)),
                 'std_mm':float(np.std(side_covers,ddof=1)) if len(side_covers)>1 else 0.},
        'modeled_top_cover_mm':mean_top_cover,
        'modeled_side_near_cover_mm':mean_side_cover,
        'modeled_bottom_cover_mm':bottom_cover,
        'modeled_side_far_cover_mm':far_cover,
    }
    if min(mean_top_cover, mean_side_cover, bottom_cover, far_cover) < 0:
        warnings.append('Shared 432x432mm section implies negative cover; review t0 and GPR depth assumptions.')
    cfg_cross = dict(cfg)
    cfg_cross.update({'bar_diameter_mm':dlong,
                      'apex_window_raw_ns':CROSS_APEX_RAW_NS,
                      'min_peak_spacing_mm':CROSS_MIN_PEAK_SPACING_MM,
                      'min_apex_score':CROSS_MIN_APEX_SCORE,
                      'min_apex_prominence':CROSS_MIN_APEX_PROMINENCE})
    cross_picks = []
    counts_cross = []
    for rec in transverse:
        mat = rec['matrix']
        face, meta, line = (rec['face'], rec['meta'], rec['line'])
        try:
            early = first_peak_time_ns(mat, meta['dt_ns'], cfg)
            if abs(early['peak_ns'] - timezero) > 0.05:
                warnings.append('%s transverse L%d: early peak %.4fns differs from t0 %.4fns by >.05ns' % (face, line, early['peak_ns'], timezero))
        except ValueError as exc:
            warnings.append('%s transverse L%d: early peak check unavailable: %s' % (face, line, exc))
        reverse = TOP_TRANSVERSE_REVERSE if face == 'top' else SIDE_TRANSVERSE_REVERSE
        alternate = TOP_TRANSVERSE_ALTERNATING if face == 'top' else SIDE_TRANSVERSE_ALTERNATING
        reverse_lines = TOP_TRANSVERSE_REVERSE_LINES if face == 'top' else SIDE_TRANSVERSE_REVERSE_LINES
        if alternate and (line - 8) % 2:
            reverse = not reverse
        if line in reverse_lines:
            reverse = not reverse
        if reverse:
            mat = mat[:, ::-1].copy()
        candidates = detect_hyperbolas(mat, meta, cfg_cross)
        half = []
        if ENABLE_HALF_HYPERBOLA_FIT:
            transverse_start = TOP_TRANSVERSE_START_MM if face == 'top' else SIDE_TRANSVERSE_START_MM
            delta = (STIRRUP_DIAMETER_MM + dlong) / 2.0
            if face == 'top':
                expected_edge = (common_Y_near + delta - transverse_start, common_Y_far - delta - transverse_start)
                expected_depth = common_Z_top + delta
            else:
                expected_edge = (common_Z_top + delta - transverse_start, common_Z_bottom - delta - transverse_start)
                expected_depth = common_Y_near + delta
            half = detect_one_sided_edge_hyperbolas(mat, meta, cfg_cross, candidates, expected_edge, expected_depth)
            for hp in half:
                candidates=[p for p in candidates if not (
                    p.get('fit_mode')=='full_hyperbola' and
                    abs(p['x_mm']-hp['x_mm'])<HALF_REPLACE_NEARBY_FULL_MM and
                    abs(p['x_mm']-hp['expected_edge_apex_mm'])>HALF_FULL_APEX_ACCEPT_TOL_MM)]
            candidates.extend(half)
        x_local = CROSS_SCAN_FIRST_X_MM + (line - 8) * meta['line_spacing_mm']
        X = x_local if face == 'top' else COLUMN_LENGTH_MM - x_local
        count = 0
        for q in candidates:
            if q.get('fit_score', -10000000000.0) < CROSS_MIN_FIT_SCORE:
                continue
            across = float(q['x_mm']) + (TOP_TRANSVERSE_START_MM if face == 'top' else SIDE_TRANSVERSE_START_MM)
            depth = float(q['depth_bar_center_mm'])
            Y, Z = (across, depth) if face == 'top' else (depth, across)
            if not (0 <= Y <= COLUMN_WIDTH_MM and 0 <= Z <= COLUMN_HEIGHT_MM):
                continue
            cross_picks.append(dict(q, file=rec['file'].name, face=face, area=face, line=line, X_mm=float(X), Y_mm=float(Y), Z_mm=float(Z)))
            count += 1
        counts_cross.append('%s L%02d transverse X=%.0f: %d independent detections' % (face, line, X, count))
    if any((p.get('fit_mode', '').startswith('one_sided') for p in cross_picks)):
        warnings.append('One-sided transverse fits use the EXPECTED edge-bar geometry as a prior; review manually.')
    if not transverse:
        warnings.append('No transverse SGY available: longitudinal-bar picks are absent (common cover still calculated).')
    unverified = [r['file'].name for r in longitudinal if r['face'] == 'side' and r['file'].name.upper().startswith('CO')]
    if unverified:
        warnings.append('Side L2-L5 short-filename mapping assumed: ' + ', '.join(unverified))
    expected_cross = {(f, l) for f in ('top', 'side') for l in range(8, 37)}
    missing_cross = sorted(expected_cross - {(r['face'], r['line']) for r in transverse})
    if missing_cross:
        warnings.append('Missing transverse lines: ' + ', '.join(('%s L%d' % x for x in missing_cross)))
    lines=[
        'GPR NEGATIVE-AMPLITUDE DETECTION — NO CLUSTERING OR RECONSTRUCTION IN PYTHON',
        'Hyperbola candidates and fits use negative signed-amplitude troughs (no Hilbert envelope).',
        'Data: '+str(root),
        'Top CSV: '+metadata_names['top']+' | Side CSV: '+metadata_names['side'],
        'Common first-negative-peak t0 = %.6f ns (8 longitudinal files)'%timezero,
        'Operational t0 only; do not interpret these points as verified depth.',
        'Epsilon_r=%.2f; v=%.3f mm/ns; antenna spacing=36mm'%(
            DIELECTRIC_EPSILON,SPEED_MM_NS/math.sqrt(DIELECTRIC_EPSILON)),
        'No +/-20mm grouping; no average stirrup X; no between-stirrup spacings.',
        'No modeled stirrups, longitudinal reinforcement lines or mirroring.',
        'TOP longitudinal raw detections: %d individual points'%len(top_long),
        'SIDE longitudinal raw detections: %d individual points'%len(side_long),
        'TOP OUTER stirrup cover mean=%.3fmm, n=%d, range=%.2f..%.2fmm'%(
            mean_top_cover,len(top_long),np.min(top_covers),np.max(top_covers)),
        'SIDE-near OUTER stirrup cover mean=%.3fmm, n=%d, range=%.2f..%.2fmm'%(
            mean_side_cover,len(side_long),np.min(side_covers),np.max(side_covers)),
        'Inferred unscanned cover: BOTTOM=%.3fmm, SIDE-far=%.3fmm'%(
            bottom_cover,far_cover),
        'Reference YZ stirrup centreline: Y=%.3f..%.3fmm, Z=%.3f..%.3fmm (432x432)'%(
            common_Y_near,common_Y_far,common_Z_top,common_Z_bottom),
        'Top transverse individual detections: %d'%sum(p['face']=='top' for p in cross_picks),
        'Side transverse individual detections: %d'%sum(p['face']=='side' for p in cross_picks),
        'One-sided transverse detections: %d'%sum(p.get('fit_mode','').startswith('one_sided') for p in cross_picks),
        'All detections represent fitted reflections; one-sided fits are constrained by expected edge positions.',
        'GLOBAL: TOP longitudinal X=x scan, Y=line, Z=depth; SIDE longitudinal X=L-x scan, Y=depth, Z=line.',
        'GLOBAL: TOP transverse X=line station, Y=across, Z=depth; SIDE transverse X=L-line station, Y=depth, Z=across.',
        '', 'LONGITUDINAL SCANS:', *counts_long,
        '', 'TRANSVERSE SCANS:', *counts_cross,
        '', 'ONE-SIDED PICKS (manual verification required):'
    ]
    lines.extend('  %s L%d local=%.1fmm, global XYZ=(%.1f,%.1f,%.1f), quality=%.2f'%(
        p['face'],p['line'],p['x_mm'],p['X_mm'],p['Y_mm'],p['Z_mm'],
        p.get('one_sided_quality',0.)) for p in cross_picks
        if p.get('fit_mode','').startswith('one_sided'))
    lines += ['', 'WARNINGS:', *warnings]
    return {
        'root':root,'cfg':cfg,'common_t0_ns':timezero,'peak_records':peaks,
        'gpr_points':long_picks,'top_gpr_points':top_long,'side_gpr_points':side_long,
        'transverse_points':cross_picks,
        'top_transverse_points':[p for p in cross_picks if p['face']=='top'],
        'side_transverse_points':[p for p in cross_picks if p['face']=='side'],
        'cover_statistics':cover_statistics,'common_section':common_section,
        'warnings':warnings,'unmapped':unmapped,'missing_cross':missing_cross,
        'log':'\n'.join(lines)
    }


# --- GRASSHOPPER OUTPUT / RHINO GEOMETRY ---
# The same nominal bar-radius offset is retained from the original method.
# It does not constitute independent validation of radar scattering centres.

def create_rhino_geometry_raw(result):
    """Make points, column and ONE YZ reference polygon; NO bar/tie geometry."""
    import Rhino.Geometry as rg

    def pt(p):
        return rg.Point3d(p['X_mm'], p['Y_mm'], p['Z_mm'])
    geo={
        'Column':rg.Box(rg.Plane.WorldXY,
                        rg.Interval(0,COLUMN_LENGTH_MM),
                        rg.Interval(0,COLUMN_WIDTH_MM),
                        rg.Interval(0,COLUMN_HEIGHT_MM)).ToBrep(),
        'GPRPoints':[pt(p) for p in result['gpr_points']],
        'TransversePoints':[pt(p) for p in result['transverse_points']],
        'TopStirrupPoints':[pt(p) for p in result['top_gpr_points']],
        'SideStirrupPoints':[pt(p) for p in result['side_gpr_points']],
        'TopLongitudinalPoints':[pt(p) for p in result['top_transverse_points']],
        'SideLongitudinalPoints':[pt(p) for p in result['side_transverse_points']],
    }
    c = result['common_section']
    y0, y1 = (c['Y_near_mm'], c['Y_far_mm'])
    z0, z1 = (c['Z_top_mm'], c['Z_bottom_mm'])
    geo['SectionYZ'] = rg.PolylineCurve([rg.Point3d(0, y0, z0), rg.Point3d(0, y1, z0), rg.Point3d(0, y1, z1), rg.Point3d(0, y0, z1), rg.Point3d(0, y0, z0)])
    return geo

OUTPUT_NAMES = (
    'Loops','Bars','Column','Points','GPRPoints','TimeZero','PeakTimes',
    'PeakInfo','Info','Longitudinal','LongitudinalPipes','LongitudinalMeasured',
    'LongitudinalInfo','TransversePoints','Log',
    'TopStirrupPoints','SideStirrupPoints','TopLongitudinalPoints',
    'SideLongitudinalPoints','CoverTop','CoverSide','CoverBottom','CoverFar',
    'SectionYZ','StirrupRecords','TransverseRecords',
)

def run_grasshopper(folder_input=None, enabled=False, pipes=False, long_d=None):
    """Retain legacy outputs as empty data where their geometry was removed."""
    values = {n: [] for n in OUTPUT_NAMES}
    values.update({'Column': None, 'TimeZero': None, 'CoverTop': None, 'CoverSide': None, 'CoverBottom': None, 'CoverFar': None, 'SectionYZ': None, 'Log': ''})
    if not enabled:
        values['Log'] = 'Set run=True to detect GPR reflections. Save the .gh beside GPR_Data.'
        return tuple((values[name] for name in OUTPUT_NAMES))
    try:
        result = process_sgy_folder(folder_input, long_d)
        geo = create_rhino_geometry_raw(result)
        values.update(geo)
        covers = result['cover_statistics']
        values.update({
            'TimeZero':result['common_t0_ns'],
            'PeakTimes':[p['peak_ns'] for p in result['peak_records']],
            'PeakInfo':['%s L%d: %.6f ns'%(r['area'],r['line'],r['peak_ns'])
                        for r in result['peak_records']],
            'Info':['RAW radar picks only — use GH for grouping/spacing/geometry.',
                    'Top cover=%.3f mm from %d points'%(covers['top']['mean_mm'],covers['top']['n']),
                    'Side cover=%.3f mm from %d points'%(covers['side']['mean_mm'],covers['side']['n']),
                    'Bottom (inferred)=%.3f mm'%covers['modeled_bottom_cover_mm'],
                    'Opposite side (inferred)=%.3f mm'%covers['modeled_side_far_cover_mm']],
            'CoverTop':covers['modeled_top_cover_mm'],
            'CoverSide':covers['modeled_side_near_cover_mm'],
            'CoverBottom':covers['modeled_bottom_cover_mm'],
            'CoverFar':covers['modeled_side_far_cover_mm'],
            'StirrupRecords':result['gpr_points'],
            'TransverseRecords':result['transverse_points'],
            'Log':result['log'],
        })
        try:
            import Rhino
            doc = Rhino.RhinoDoc.ActiveDoc
            if doc is not None and str(doc.ModelUnitSystem).lower() != 'millimeters':
                values['Log'] += '\nWARNING: Rhino units are not millimeters.'
        except Exception:
            pass
    except Exception:
        import traceback
        err = 'ERROR during GPR raw detection:\n' + traceback.format_exc()
        values['Log'] = err
        try:
            import Grasshopper
            ghenv.Component.AddRuntimeMessage(Grasshopper.Kernel.GH_RuntimeMessageLevel.Error, err[-1800:])
        except Exception:
            pass
    return tuple((values[name] for name in OUTPUT_NAMES))

if 'run' in globals() or 'folder' in globals():
    _gpr_out = run_grasshopper(globals().get('folder'), bool(globals().get('run', False)), bool(globals().get('makePipes', False)), globals().get('longDiameter'))
    (Loops,Bars,Column,Points,GPRPoints,TimeZero,PeakTimes,PeakInfo,Info,
     Longitudinal,LongitudinalPipes,LongitudinalMeasured,LongitudinalInfo,
     TransversePoints,Log,TopStirrupPoints,SideStirrupPoints,
     TopLongitudinalPoints,SideLongitudinalPoints,CoverTop,CoverSide,
     CoverBottom,CoverFar,SectionYZ,StirrupRecords,TransverseRecords) = _gpr_out
    out = Log

if __name__ == '__main__' and 'run' not in globals() and ('folder' not in globals()):
    import argparse
    arg = argparse.ArgumentParser(description='Read Proceq SGY; export every ungrouped fitted GPR reflection.')
    arg.add_argument('folder', nargs='?', default=None)
    arg.add_argument('--long-diameter', type=float, default=None)
    args = arg.parse_args()
    r = process_sgy_folder(args.folder, args.long_diameter)
    print(r['log'])
