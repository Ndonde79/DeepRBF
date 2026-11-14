# DeepRBF - Copyright (c) 2026 DeepFusion Geoscience Limited. Licensed under the BSD 3-Clause License (see LICENSE).
"""End-to-end shell workflow on synthetic drillholes: CSV intake, minimum-curvature desurvey, compositing, indicator RBF
(spherical, drift none), topography clipping from a DXF surface, small-body filter, Surpac/DXF export, reference
comparison and withheld-hole folds. A dipping slab of grade must be recovered and clipped under the topography."""
import json, os, sys, numpy as np, pandas as pd, pytest
sys.path.insert(0, os.environ.get('DEEPRBF_MODULE_DIR', ''))
pytest.importorskip('skimage')
from deeprbf.workflow import shell, mesh, surfaces

def synthetic(tmp):
    r = np.random.default_rng(0); holes = []; surveys = []; rows = []
    for i, (x, y) in enumerate([(x, y) for x in np.arange(0, 200, 20.0) for y in np.arange(0, 200, 20.0)]):
        h = f'H{i:03d}'; z = 100.0; dip = 90.0 if i % 3 else 60.0; az = 270.0
        holes.append((h, x + r.uniform(-2, 2), y + r.uniform(-2, 2), z)); surveys.append((h, 0.0, dip, az)); surveys.append((h, 80.0, dip, az))
        for d in np.arange(0, 80, 1.0):
            t = np.radians(dip); px = holes[-1][1] - np.sin(np.radians(az)) * 0 + np.sin(np.radians(az)) * np.cos(t) * (d + .5); pz = z - np.sin(t) * (d + .5)
            slab = abs((pz - 60) - 0.2 * (px - 100)) < 6                      # slab dipping gently toward +x
            rows.append((h, d, d + 1, (1.5 if slab else 0.05) * r.lognormal(0, 0.3)))
    pd.DataFrame(holes, columns=['BHID', 'XCOLLAR', 'YCOLLAR', 'ZCOLLAR']).to_csv(tmp / 'collar.csv', index=False)
    pd.DataFrame(surveys, columns=['BHID', 'AT', 'DIP', 'BRG']).to_csv(tmp / 'survey.csv', index=False)
    pd.DataFrame(rows, columns=['BHID', 'FROM', 'TO', 'AU']).to_csv(tmp / 'assay.csv', index=False)
    V = np.array([[-50, -50, 75.0], [250, -50, 75.0], [-50, 250, 75.0], [250, 250, 75.0]]); mesh.write_dxf(V, np.array([[0, 1, 2], [1, 3, 2]]), str(tmp / 'topo.dxf'))
    return dict(name='synthetic', drillholes=dict(collar='collar.csv', survey='survey.csv', intervals='assay.csv', value='AU', composite_length=2.0),
                cutoff=0.5, trend=dict(dip=11.3, dip_azimuth=90.0, pitch=90, ranges=[60.0, 60.0, 12.0]),
                interpolant=dict(kernel='spherical', sill=1.0, nugget=0.02, drift='none'),
                surface=dict(iso=0.4, spacing=4.0, min_volume=500.0, padding=10.0, topography='topo.dxf'),
                validation=dict(folds=3), output_dir='out')

def test_shell_end_to_end(tmp_path):
    cfg = synthetic(tmp_path); json.dump(cfg, open(tmp_path / 's.json', 'w'))
    assert shell.main([str(tmp_path / 's.json')]) == 0
    rep = json.load(open(tmp_path / 'out' / 'synthetic_report.json'))
    assert rep['drillholes']['holes'] == 100 and rep['drillholes']['composites'] == 4000
    assert rep['honouring']['f1'] > 0.9 and rep['withheld_holes']['f1'] > rep['withheld_holes']['baseline_f1_all_ore']
    for f in rep['wireframe']['files']: assert os.path.getsize(f) > 0
    T = mesh.read_wireframe(str(tmp_path / 'out' / 'synthetic.dtm')); assert T.reshape(-1, 3)[:, 2].max() <= 75.0 + 4.0   # clipped under topography
    assert np.allclose(T, mesh.read_wireframe(str(tmp_path / 'out' / 'synthetic.dxf')), atol=1e-3)                       # Surpac and DXF agree
    cfg['validation'] = dict(folds=0, reference='out/synthetic.dtm'); cfg['output_dir'] = 'out2'; rep2 = shell.run(cfg, str(tmp_path))
    assert rep2['reference']['shared_share_of_reference'] > 0.90                                                         # same settings reproduce the shell

def test_surface_formats(tmp_path):
    V = np.array([[0, 0, 1.0], [5, 0, 2.0], [0, 5, 3.0]]); mesh.write_surpac(V, np.array([[0, 1, 2]]), str(tmp_path / 't'))
    assert len(surfaces.read_points(str(tmp_path / 't.dtm'))) == 3
    pd.DataFrame(V, columns=['XP', 'YP', 'ZP']).to_csv(tmp_path / 'p.csv', index=False); assert np.allclose(surfaces.read_points(str(tmp_path / 'p.csv')), V)
    with pytest.raises(ValueError): surfaces.read_points(str(tmp_path / 'x.dm'))
