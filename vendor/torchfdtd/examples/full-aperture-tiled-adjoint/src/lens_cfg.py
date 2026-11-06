"""Lens and aperture geometries of the examples (FF_LENS); d6 is a small test fixture."""
import os,json
from pathlib import Path
LENSES={
 'd200':dict(name='d200',D_UM=200.,R_AP=100.,F_UM=200./.6,OBJ_DIST_UM=0.,PUPIL_GRID=800,PITCH_UM=.25,NPX=10000,XG0=-100.),
 'd12':dict(name='d12',D_UM=12.,R_AP=6.,F_UM=12./.6,OBJ_DIST_UM=0.,PUPIL_GRID=48,PITCH_UM=.25,NPX=600,XG0=-6.),
 'd115s':dict(name='d115s',D_UM=200.*540./940.,R_AP=100.*540./940.,F_UM=200.*540./940./.6,OBJ_DIST_UM=0.,PUPIL_GRID=464,PITCH_UM=.25,NPX=5800,XG0=-58.,
              SCALE=940./540.,PHYSICAL_D_UM=200.,PHYSICAL_LAMBDA_NM=940.),   # 2026-09-30: E3 at 940 nm as the 540 nm model scaled by 1/s (grid 116 um)
 's150':dict(name='s150',D_UM=150.,R_AP=75.,SHAPE='square',F_UM=300.,OBJ_DIST_UM=0.,PUPIL_GRID=600,PITCH_UM=.25,NPX=7500,XG0=-75.),   # 2026-09-30: E2 150 x 150 um square (R_AP = half-width)
 's100':dict(name='s100',D_UM=100.,R_AP=50.,SHAPE='square',F_UM=200.,OBJ_DIST_UM=0.,PUPIL_GRID=400,PITCH_UM=.25,NPX=5000,XG0=-50.),   # 2026-10-01: E3 100 x 100 um square (R_AP = half-width)
 'd6':dict(name='d6',D_UM=6.,R_AP=3.,F_UM=6./.6,OBJ_DIST_UM=0.,PUPIL_GRID=24,PITCH_UM=.25,NPX=300,XG0=-3.),   # 2026-09-30: local smoke fixture
}
def get():return dict(LENSES[os.environ.get('FF_LENS','d200')])
def write_run_json(out_dir,L=None):
 L=get() if L is None else L
 Path(out_dir,'geometry.json').write_text(json.dumps(dict(L,H10=L['NPX'],latent_pitch_um=.01,fdtd_pitch_um=.02),indent=2))
