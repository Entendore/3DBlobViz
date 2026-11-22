"""
3D Blob Visualizer (Ultimate Edition)
=====================================
Features: Multithreaded marching cubes, VBO caching, dynamic gravity physics, 
fake bloom, ambient particles, shockwaves, OBJ/PNG export, and granular animation controls.

Dependencies: PySide6, pyqtgraph (with OpenGL), numpy, scikit-image
    pip install PySide6 pyqtgraph PyOpenGL numpy scikit-image
"""

import sys
import time
import numpy as np
from dataclasses import dataclass, field

from PySide6.QtCore import Qt, QTimer, Signal, QThread, QObject
from PySide6.QtGui import QColor, QFont, QKeyEvent
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QCheckBox, QGroupBox, QSpinBox,
    QColorDialog, QFrame, QButtonGroup, QFileDialog
)

import pyqtgraph as pg
import pyqtgraph.opengl as gl
from skimage.measure import marching_cubes

# ---------- Constants & Determinism -------------------------------------------
SEED = 42
np.random.seed(SEED)

GRID_BOUNDS = 3.0
GRID_DEFAULT = 48

# ---------- Stylesheet --------------------------------------------------------
STYLE = """
QMainWindow, QWidget#central { background-color: #0e1116; }
QLabel { color: #c8d3e0; font-size: 12px; }
QLabel#title    { color: #ffffff; font-size: 15px; font-weight: 700; letter-spacing: 2px; }
QLabel#subtitle { color: #56b6c2; font-size: 10px; letter-spacing: 1px; }
QLabel#hint     { color: #4a5666; font-size: 10px; }
QLabel#value    { color: #56b6c2; font-size: 11px;
                  font-family: 'Consolas','Courier New',monospace; }
QGroupBox {
    color: #8aa0b8; border: 1px solid #1f2733; border-radius: 6px;
    margin-top: 14px; padding: 14px 10px 10px 10px; font-size: 11px;
    font-weight: bold; background-color: #131820; letter-spacing: 1px;
}
QGroupBox::title {
    subcontrol-origin: margin; subcontrol-position: top left;
    left: 10px; padding: 0 5px; background-color: #0e1116;
}
QPushButton {
    background-color: #1f2733; color: #c8d3e0; border: 1px solid #2a3441;
    border-radius: 4px; padding: 6px 12px; font-size: 11px;
}
QPushButton:hover    { background-color: #2a3441; border-color: #56b6c2; }
QPushButton:pressed  { background-color: #131820; }
QPushButton:checked  {
    background-color: #56b6c2; color: #0e1116;
    border-color: #56b6c2; font-weight: bold;
}
QSlider::groove:horizontal { height: 3px; background: #1f2733; border-radius: 2px; }
QSlider::handle:horizontal {
    background: #56b6c2; width: 12px; height: 12px;
    margin: -5px 0; border-radius: 6px;
}
QSlider::handle:horizontal:hover { background: #6ec8d3; }
QSlider::sub-page:horizontal { background: #56b6c2; border-radius: 2px; }
QCheckBox { color: #c8d3e0; spacing: 8px; font-size: 11px; }
QCheckBox::indicator {
    width: 14px; height: 14px; border: 1px solid #2a3441;
    border-radius: 3px; background: #1f2733;
}
QCheckBox::indicator:checked { background: #56b6c2; border-color: #56b6c2; }
QSpinBox {
    background-color: #1f2733; color: #c8d3e0; border: 1px solid #2a3441;
    border-radius: 3px; padding: 2px 4px; font-size: 11px; min-width: 60px;
}
QStatusBar { background-color: #0a0d12; color: #56b6c2; }
QStatusBar::item { border: none; }
"""

# ---------- Model -------------------------------------------------------------
@dataclass(slots=True)
class Blob:
    freq: np.ndarray
    phase: np.ndarray
    size_factor: float
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    def kinematic_pos(self, t: float, amp: float, complexity: float, vert_scale: float, turbulence: float, flow: float, center_pull: float, phase_drift: float) -> np.ndarray:
        # Time warping for rhythmic speed-up/slow-down
        t_eff = t + np.sin(t * 0.5) * flow
        
        # Gradually shifting phases so trajectories never truly repeat
        effective_phase = self.phase + t * phase_drift
        
        # Base Lissajous trajectory
        freqs = self.freq * complexity
        pos = amp * np.sin(freqs * t_eff + effective_phase)
        
        # Scale vertical axis
        pos[2] *= vert_scale
        
        # Add layered high-frequency noise for organic turbulence
        if turbulence > 0:
            noise = np.sin(t_eff * 5.1 + self.phase[0] * 2) * 0.5
            noise += np.sin(t_eff * 7.3 + self.phase[1] * 3) * 0.5
            pos += noise * turbulence
            
        # Center pull (gravitate toward origin)
        if center_pull > 0:
            pos *= (1.0 - center_pull * 0.8)
            
        return pos

def apply_colormap(t: np.ndarray, mode: str):
    t = np.clip(t, 0.0, 1.0)
    if mode == "aurora":
        r = 0.10 + 0.55 * t ** 1.6
        g = 0.55 + 0.20 * np.sin(t * np.pi) - 0.30 * t
        b = 0.75 + 0.20 * np.sin(t * np.pi * 0.7 + 0.5)
    elif mode == "magma":
        r = 0.10 + 0.90 * t ** 0.6
        g = 0.02 + 0.85 * t ** 2.6
        b = 0.15 + 0.40 * (1.0 - t) ** 1.8 + 0.35 * t ** 3.5
    elif mode == "ocean":
        r = 0.05 + 0.60 * t ** 2.0
        g = 0.20 + 0.70 * t ** 1.1
        b = 0.55 + 0.40 * t ** 0.8
    elif mode == "neon":
        r = 0.1 + 0.9 * t
        g = 0.1 + 0.9 * np.sin(t * np.pi)
        b = 0.8 - 0.4 * t
    elif mode == "toxic":
        r = 0.3 + 0.6 * np.sin(t * np.pi * 2)
        g = 0.8 + 0.2 * t
        b = 0.2 + 0.5 * (1 - t)
    else:
        r = g = b = np.full_like(t, 0.5)
    return r, g, b

class MeshWorker(QObject):
    mesh_ready = Signal(object, object, object, object, object, object)
    request_work = Signal(object, object, float, int, str)

    def __init__(self, bounds: float):
        super().__init__()
        self.bounds = bounds
        self.N = GRID_DEFAULT
        xs = np.linspace(-bounds, bounds, self.N, dtype=np.float32)
        self.X, self.Y, self.Z = np.meshgrid(xs, xs, xs, indexing="ij")
        self.request_work.connect(self._process)

    def _update_grid(self, N: int):
        if N == self.N: return
        self.N = N
        xs = np.linspace(-self.bounds, self.bounds, self.N, dtype=np.float32)
        self.X, self.Y, self.Z = np.meshgrid(xs, xs, xs, indexing="ij")

    def _process(self, centers, radii, threshold, grid_size, color_mode):
        self._update_grid(grid_size)
        field = np.zeros(self.X.shape, dtype=np.float32)
        for c, r in zip(centers, radii):
            dx = self.X - c[0]; dy = self.Y - c[1]; dz = self.Z - c[2]
            d2 = dx * dx + dy * dy + dz * dz
            field += 1.0 / (1.0 + d2 / (r * r))

        try:
            verts, faces, _, _ = marching_cubes(
                field, level=threshold,
                spacing=(2.0 * self.bounds / (self.N - 1),) * 3,
                allow_degenerate=False
            )
            verts = verts.astype(np.float32) - self.bounds
            faces = faces.astype(np.uint32)

            y = verts[:, 1]
            y_min, y_max = float(y.min()), float(y.max())
            span = max(y_max - y_min, 1e-6)
            t = (y - y_min) / span
            r, g, b = apply_colormap(t, color_mode)
            colors = np.zeros((len(verts), 4), dtype=np.float32)
            colors[:, 0] = r; colors[:, 1] = g; colors[:, 2] = b; colors[:, 3] = 1.0

            shadow_verts = verts.copy()
            shadow_verts[:, 2] = -self.bounds - 0.05
            shadow_colors = np.zeros((len(shadow_verts), 4), dtype=np.float32)
            shadow_colors[:, 3] = 0.35

            self.mesh_ready.emit(verts, faces, colors, shadow_verts, faces, shadow_colors)
        except (RuntimeError, ValueError):
            self.mesh_ready.emit(None, None, None, None, None, None)

# ---------- View / Controller -------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("◆ 3D Blob Visualizer (Ultimate)")
        self.resize(1400, 860)

        # State
        self.time = 0.0
        self.threshold = 0.5
        self.grid_size = GRID_DEFAULT
        self.amplitude = 2.0
        self.base_radius = 0.8
        self.speed = 1.0
        self.pulse_strength = 0.0
        
        # Animation States
        self.complexity = 1.0
        self.vert_scale = 1.0
        self.turbulence = 0.0
        self.flow = 0.0
        self.center_pull = 0.0
        self.phase_drift = 0.0
        self.rotation_speed = 0.3
        
        self.paused = False
        self.auto_rotate = True
        self.wireframe = False
        self.is_physics = False
        self.glow_enabled = True
        self.color_mode = "aurora"
        self.bg_color = QColor(14, 17, 22)
        
        # Worker Thread Setup
        self.worker_busy = False
        self.worker = MeshWorker(GRID_BOUNDS)
        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)
        self.worker.mesh_ready.connect(self._on_mesh_ready)
        self.worker_thread.start()

        self.blobs = self._create_blobs(6)
        self.fps_history: list[float] = []
        self.last_time = time.perf_counter()

        self._build_ui()
        self._setup_view()

        # Pre-create mesh items (VBO Caching)
        self.mesh_item = gl.GLMeshItem(
            vertexes=np.array([[0,0,0]], dtype=np.float32),
            faces=np.array([[0,0,0]], dtype=np.uint32),
            smooth=True, shader="shaded", glOptions="opaque",
            drawFaces=True, drawEdges=False
        )
        self.glow_item = gl.GLMeshItem(
            vertexes=np.array([[0,0,0]], dtype=np.float32),
            faces=np.array([[0,0,0]], dtype=np.uint32),
            smooth=True, glOptions="additive",
            drawFaces=True, drawEdges=False, shader="normalColor"
        )
        self.shadow_item = gl.GLMeshItem(
            vertexes=np.array([[0,0,0]], dtype=np.float32),
            faces=np.array([[0,0,0]], dtype=np.uint32),
            smooth=True, glOptions="translucent",
            drawFaces=True, drawEdges=False, color=(0,0,0,0.3)
        )
        self.view.addItem(self.shadow_item)
        self.view.addItem(self.glow_item)
        self.view.addItem(self.mesh_item)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(16)  # ~60 FPS target

    def _create_blobs(self, n: int) -> list[Blob]:
        rng = np.random.default_rng(SEED)
        out: list[Blob] = []
        for _ in range(n):
            freq = rng.uniform(0.4, 1.2, size=3).astype(np.float32)
            phase = rng.uniform(0.0, 2.0 * np.pi, size=3).astype(np.float32)
            size = float(rng.uniform(0.7, 1.3))
            out.append(Blob(freq=freq, phase=phase, size_factor=size))
        return out

    def _build_ui(self):
        central = QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_panel(), 0)

        view_frame = QFrame()
        view_frame.setStyleSheet("background-color: #0e1116;")
        vl = QVBoxLayout(view_frame)
        vl.setContentsMargins(6, 6, 6, 6)
        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor(self.bg_color)
        self.view.setCameraPosition(distance=11, elevation=18, azimuth=35)
        vl.addWidget(self.view)
        root.addWidget(view_frame, 1)

        self.status_info = QLabel("Ready")
        self.status_fps = QLabel("FPS: --")
        self.statusBar().addWidget(self.status_info)
        self.statusBar().addPermanentWidget(self.status_fps)

    def _build_panel(self) -> QFrame:
        panel = QFrame()
        panel.setFixedWidth(340) # Widened slightly to fit 2-column slider labels nicely
        panel.setStyleSheet("QFrame { background-color: #0e1116; border-right: 1px solid #1f2733; }")
        L = QVBoxLayout(panel)
        L.setContentsMargins(16, 16, 16, 16)
        L.setSpacing(10)

        title = QLabel("◆ BLOB VISUALIZER"); title.setObjectName("title")
        sub = QLabel("ULTIMATE METABALL ENGINE"); sub.setObjectName("subtitle")
        L.addWidget(title); L.addWidget(sub)

        # Animation Group
        anim = QGroupBox("ANIMATION & TRAJECTORY")
        ag = QVBoxLayout(anim); ag.setSpacing(8); ag.setContentsMargins(10, 18, 10, 10)

        self.speed_slider,  r1 = self._slider("Speed",       0, 300, 100, lambda v: f"{v/100:.2f}×")
        self.count_slider,  r2 = self._slider("Blob count",  1,  16,   6, lambda v: f"{v}")
        self.radius_slider, r3 = self._slider("Blob radius", 20,200,  80, lambda v: f"{v/100:.2f}")
        self.amp_slider,    r4 = self._slider("Spread",      50,300, 200, lambda v: f"{v/100:.2f}")
        self.thr_slider,    r5 = self._slider("Threshold",   10,100,  50, lambda v: f"{v/100:.2f}")
        self.pulse_slider,  r6 = self._slider("Pulse",       0, 100,   0, lambda v: f"{v/100:.2f}")
        
        self.comp_slider,   r7 = self._slider("Complexity",  20,300, 100, lambda v: f"{v/100:.2f}×")
        self.vert_slider,   r8 = self._slider("Verticality",  0,200, 100, lambda v: f"{v/100:.2f}×")
        self.turb_slider,   r9 = self._slider("Turbulence",   0,200,   0, lambda v: f"{v/100:.2f}")
        self.flow_slider,  r10 = self._slider("Time Flow",    0,100,   0, lambda v: f"{v/100:.2f}")
        self.pull_slider,  r11 = self._slider("Center Pull",  0,100,   0, lambda v: f"{v/100:.2f}")
        self.drift_slider, r12 = self._slider("Phase Drift",  0,100,   0, lambda v: f"{v/100:.2f}")
        
        # Camera Rotation Speed slider
        self.cam_rot_slider, r13 = self._slider("Cam Rotate", 0, 200, 30, lambda v: f"{v/100:.2f}°/f")
        
        self.speed_slider.valueChanged.connect(self._on_speed)
        self.count_slider.valueChanged.connect(self._on_count)
        self.radius_slider.valueChanged.connect(self._on_radius)
        self.amp_slider.valueChanged.connect(self._on_amplitude)
        self.thr_slider.valueChanged.connect(self._on_threshold)
        self.pulse_slider.valueChanged.connect(self._on_pulse)
        
        self.comp_slider.valueChanged.connect(self._on_complexity)
        self.vert_slider.valueChanged.connect(self._on_vert_scale)
        self.turb_slider.valueChanged.connect(self._on_turbulence)
        self.flow_slider.valueChanged.connect(self._on_flow)
        self.pull_slider.valueChanged.connect(self._on_center_pull)
        self.drift_slider.valueChanged.connect(self._on_phase_drift)
        self.cam_rot_slider.valueChanged.connect(self._on_cam_rotate)
        
        for r in (r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13): ag.addLayout(r)
        L.addWidget(anim)

        # Environment & Physics
        env = QGroupBox("ENVIRONMENT")
        eg = QVBoxLayout(env); eg.setSpacing(6); eg.setContentsMargins(10, 18, 10, 10)
        
        self.physics_check = QCheckBox("Gravity Physics Mode")
        self.physics_check.setToolTip("Toggle between mathematical paths and live physics simulation")
        self.physics_check.toggled.connect(self._on_physics)
        eg.addWidget(self.physics_check)

        self.particles_check = QCheckBox("Ambient Particles")
        self.particles_check.setChecked(True)
        self.particles_check.toggled.connect(self._on_particles)
        eg.addWidget(self.particles_check)

        self.grid_check = QCheckBox("Show floor grid")
        self.grid_check.setChecked(True)
        self.grid_check.toggled.connect(self._on_grid_toggle)
        eg.addWidget(self.grid_check)

        grow = QHBoxLayout()
        grow.addWidget(QLabel("Grid resolution"))
        grow.addStretch(1)
        self.grid_spin = QSpinBox()
        self.grid_spin.setRange(16, 128)
        self.grid_spin.setSingleStep(8)
        self.grid_spin.setValue(GRID_DEFAULT)
        self.grid_spin.setSuffix("³")
        self.grid_spin.valueChanged.connect(self._on_grid_size)
        grow.addWidget(self.grid_spin)
        eg.addLayout(grow)
        L.addWidget(env)

        # Visuals
        vis = QGroupBox("VISUALS")
        vg = QVBoxLayout(vis); vg.setSpacing(6); vg.setContentsMargins(10, 18, 10, 10)

        self.wireframe_check = QCheckBox("Wireframe mode")
        self.wireframe_check.toggled.connect(self._on_wireframe)
        vg.addWidget(self.wireframe_check)

        self.glow_check = QCheckBox("Bloom / Glow effect")
        self.glow_check.setChecked(True)
        self.glow_check.toggled.connect(self._on_glow)
        vg.addWidget(self.glow_check)

        crow = QHBoxLayout()
        crow.addWidget(QLabel("Colors:"))
        crow.addStretch(1)
        self.color_btns = QButtonGroup(self)
        self.color_btns.setExclusive(True)
        for name, mode in [("Aurora", "aurora"), ("Magma", "magma"), ("Ocean", "ocean"), ("Neon", "neon"), ("Toxic", "toxic")]:
            b = QPushButton(name)
            b.setCheckable(True)
            b.setFixedHeight(24)
            if mode == "aurora": b.setChecked(True)
            self.color_btns.addButton(b)
            b.clicked.connect(lambda _=False, m=mode: self._on_color_mode(m))
            crow.addWidget(b)
        vg.addLayout(crow)

        bg_btn = QPushButton("Pick Background Color…")
        bg_btn.clicked.connect(self._pick_bg)
        vg.addWidget(bg_btn)
        L.addWidget(vis)

        # Export
        exp = QGroupBox("EXPORT")
        xg = QHBoxLayout(exp); xg.setSpacing(6); xg.setContentsMargins(10, 18, 10, 10)
        obj_btn = QPushButton("💾 Save .OBJ")
        obj_btn.setToolTip("Export the current 3D mesh to an OBJ file")
        obj_btn.clicked.connect(self._export_obj)
        xg.addWidget(obj_btn)

        png_btn = QPushButton("📸 Screenshot")
        png_btn.setToolTip("Save a high-resolution PNG of the viewport")
        png_btn.clicked.connect(self._screenshot)
        xg.addWidget(png_btn)
        L.addWidget(exp)

        # Actions
        act = QHBoxLayout()
        shock_btn = QPushButton("💥 Shockwave")
        shock_btn.setStyleSheet("QPushButton { background-color: #3a1f27; border-color: #c25662; } QPushButton:hover { background-color: #4a2733; }")
        shock_btn.clicked.connect(self._trigger_shockwave)
        act.addWidget(shock_btn)
        
        reset_btn = QPushButton("↺ Reset")
        reset_btn.clicked.connect(self._reset)
        act.addWidget(reset_btn)
        L.addLayout(act)

        L.addStretch(1)

        hint = QLabel("DRAG to orbit · SCROLL to zoom\nSPACE pause · R reset · ESC quit")
        hint.setObjectName("hint")
        hint.setAlignment(Qt.AlignCenter)
        L.addWidget(hint)
        return panel

    def _slider(self, name, lo, hi, default, fmt):
        outer = QVBoxLayout(); outer.setSpacing(2)
        head = QHBoxLayout()
        lbl = QLabel(name)
        val = QLabel(fmt(default)); val.setObjectName("value"); val.setAlignment(Qt.AlignRight)
        head.addWidget(lbl); head.addWidget(val); outer.addLayout(head)
        s = QSlider(Qt.Horizontal); s.setRange(lo, hi); s.setValue(default)
        s.valueChanged.connect(lambda v: val.setText(fmt(v)))
        outer.addWidget(s)
        return s, outer

    def _setup_view(self):
        self.grid_item = gl.GLGridItem()
        self.grid_item.setSize(14, 14)
        self.grid_item.setSpacing(1, 1)
        self.grid_item.setColor(QColor(40, 55, 75, 180))
        self.grid_item.translate(0, 0, -GRID_BOUNDS - 0.1)
        self.view.addItem(self.grid_item)

        # Setup Particles
        n_particles = 300
        self.particle_pos = np.random.uniform(-GRID_BOUNDS, GRID_BOUNDS, (n_particles, 3)).astype(np.float32)
        self.particle_vel = np.zeros_like(self.particle_pos)
        self.particle_vel[:, 2] = np.random.uniform(0.2, 0.8, n_particles).astype(np.float32)
        self.particle_item = gl.GLScatterPlotItem(
            pos=self.particle_pos, color=(1, 1, 1, 0.3), size=2.5, pxMode=True
        )
        self.view.addItem(self.particle_item)

        # Setup Shockwave Ring
        self.shockwaves = []
        self.shock_ring_pts = np.array([[np.cos(a), np.sin(a), 0] for a in np.linspace(0, 2*np.pi, 64)], dtype=np.float32)
        self.shockwave_item = gl.GLLinePlotItem(pos=np.zeros((2,3)), color=(1,0,0,0), width=2, mode='line_strip', antialias=True)
        self.view.addItem(self.shockwave_item)

    # ---- main tick & threading ---------------------------------------------
    def _tick(self):
        now = time.perf_counter()
        dt = now - self.last_time
        self.last_time = now
        if dt > 0:
            self.fps_history.append(1.0 / dt)
            if len(self.fps_history) > 30: self.fps_history.pop(0)
            avg = sum(self.fps_history) / len(self.fps_history)
            self.status_fps.setText(f"FPS: {avg:5.1f}")

        if not self.paused:
            self.time += 0.016 * self.speed
            self._update_particles(dt)
            self._update_shockwaves(dt)
            self._request_mesh_update()

        if self.auto_rotate and not self.paused:
            az = self.view.opts["azimuth"]
            self.view.setCameraPosition(azimuth=(az + self.rotation_speed) % 360)

    def _update_particles(self, dt):
        self.particle_pos += self.particle_vel * dt * self.speed
        mask = self.particle_pos[:, 2] > GRID_BOUNDS
        self.particle_pos[mask, 2] = -GRID_BOUNDS
        self.particle_item.setData(pos=self.particle_pos)

    def _trigger_shockwave(self):
        self.shockwaves.append({'t': 0.0, 'pos': np.array([0,0,0], dtype=np.float32)})
        if self.is_physics:
            for b in self.blobs:
                b.vel[2] += 5.0

    def _update_shockwaves(self, dt):
        if not self.shockwaves:
            if self.shockwave_item.color is not None:
                self.shockwave_item.setData(color=(1,0,0,0))
            return
        
        active = []
        for sw in self.shockwaves:
            sw['t'] += dt * self.speed
            if sw['t'] < 1.5:
                active.append(sw)
        self.shockwaves = active
        
        if self.shockwaves:
            sw = self.shockwaves[-1]
            r = sw['t'] * 4.0
            alpha = max(0.0, 1.0 - sw['t'] / 1.5)
            pts = self.shock_ring_pts * r + sw['pos']
            self.shockwave_item.setData(pos=pts, color=(1, 0.8, 0.4, alpha), width=3.0)

    def _request_mesh_update(self):
        if self.worker_busy: return
        self.worker_busy = True
        
        if self.is_physics:
            gravity = np.array([0, 0, -4.0], dtype=np.float32)
            restitution = 0.6
            dt = 0.016 * self.speed
            centers = np.zeros((len(self.blobs), 3), dtype=np.float32)
            for i, b in enumerate(self.blobs):
                b.vel += gravity * dt
                b.pos += b.vel * dt
                
                r = b.size_factor * self.base_radius
                if b.pos[2] < -GRID_BOUNDS + r:
                    b.pos[2] = -GRID_BOUNDS + r
                    b.vel[2] = -b.vel[2] * restitution
                
                for ax in range(3):
                    if abs(b.pos[ax]) > GRID_BOUNDS - r:
                        b.pos[ax] = np.sign(b.pos[ax]) * (GRID_BOUNDS - r)
                        b.vel[ax] = -b.vel[ax] * restitution
                
                centers[i] = b.pos
        else:
            centers = np.array([
                b.kinematic_pos(self.time, self.amplitude, self.complexity, self.vert_scale, self.turbulence, self.flow, self.center_pull, self.phase_drift) 
                for b in self.blobs
            ])

        pulse = 1.0 + np.sin(self.time * 3.0) * self.pulse_strength
        radii = np.array([b.size_factor * self.base_radius * pulse for b in self.blobs], dtype=np.float32)

        self.worker.request_work.emit(centers, radii, self.threshold, self.grid_size, self.color_mode)

    def _on_mesh_ready(self, verts, faces, colors, s_verts, s_faces, s_colors):
        self.worker_busy = False
        
        if verts is None or len(verts) == 0:
            self.mesh_item.setVisible(False); self.glow_item.setVisible(False); self.shadow_item.setVisible(False)
            self.status_info.setText(f"Blobs: {len(self.blobs)} | Grid: {self.grid_size}³ | Verts: 0")
            return

        mesh_data = gl.MeshData(vertexes=verts, faces=faces, vertexColors=colors)
        self.mesh_item.setMeshData(meshdata=mesh_data)
        self.mesh_item.setDrawFaces(not self.wireframe)
        self.mesh_item.setDrawEdges(self.wireframe)
        self.mesh_item.setVisible(True)

        if self.glow_enabled and not self.wireframe:
            glow_verts = verts * 1.08
            glow_colors = colors.copy()
            glow_colors[:, 3] = 0.15 
            glow_mesh_data = gl.MeshData(vertexes=glow_verts, faces=faces, vertexColors=glow_colors)
            self.glow_item.setMeshData(meshdata=glow_mesh_data)
            self.glow_item.setVisible(True)
        else:
            self.glow_item.setVisible(False)

        if not self.wireframe:
            s_mesh_data = gl.MeshData(vertexes=s_verts, faces=s_faces, vertexColors=s_colors)
            self.shadow_item.setMeshData(meshdata=s_mesh_data)
            self.shadow_item.setVisible(True)
        else:
            self.shadow_item.setVisible(False)

        self.status_info.setText(f"Blobs: {len(self.blobs)} | Grid: {self.grid_size}³ | Verts: {len(verts)}")

    # ---- handlers -----------------------------------------------------------
    def _on_speed(self, v):    self.speed = v / 100.0
    def _on_radius(self, v):   self.base_radius = v / 100.0
    def _on_amplitude(self, v):self.amplitude = v / 100.0
    def _on_pulse(self, v):    self.pulse_strength = v / 100.0
    
    # Animation Handlers
    def _on_complexity(self, v): self.complexity = v / 100.0
    def _on_vert_scale(self, v): self.vert_scale = v / 100.0
    def _on_turbulence(self, v): self.turbulence = v / 100.0
    def _on_flow(self, v):       self.flow = v / 100.0
    def _on_center_pull(self, v):self.center_pull = v / 100.0
    def _on_phase_drift(self, v):self.phase_drift = v / 100.0
    def _on_cam_rotate(self, v): self.rotation_speed = v / 100.0
    
    def _on_grid_toggle(self, c): self.grid_item.setVisible(c)
    def _on_particles(self, c): self.particle_item.setVisible(c)
    def _on_glow(self, c):     self.glow_enabled = c
    
    def _on_physics(self, c):
        self.is_physics = c
        if c:
            for b in self.blobs:
                b.pos = b.kinematic_pos(self.time, self.amplitude, self.complexity, self.vert_scale, self.turbulence, self.flow, self.center_pull, self.phase_drift)
                b.vel = np.zeros(3, dtype=np.float32)

    def _on_threshold(self, v): self.threshold = v / 100.0

    def _on_count(self, n):
        if n > len(self.blobs):
            rng = np.random.default_rng(SEED + n * 7)
            while len(self.blobs) < n:
                freq = rng.uniform(0.4, 1.2, size=3).astype(np.float32)
                phase = rng.uniform(0.0, 2.0 * np.pi, size=3).astype(np.float32)
                size = float(rng.uniform(0.7, 1.3))
                self.blobs.append(Blob(freq=freq, phase=phase, size_factor=size))
        else:
            self.blobs = self.blobs[:n]

    def _on_grid_size(self, v): self.grid_size = v
    def _on_wireframe(self, c): self.wireframe = c
    def _on_color_mode(self, mode): self.color_mode = mode

    def _pick_bg(self):
        c = QColorDialog.getColor(self.bg_color, self, "Background color")
        if c.isValid():
            self.bg_color = c
            self.view.setBackgroundColor(c)

    def _reset(self):
        self.time = 0.0
        self.blobs = self._create_blobs(len(self.blobs))
        self.shockwaves.clear()

    def _export_obj(self):
        if self.mesh_item.meshData() is None or len(self.mesh_item.meshData().vertexes()) == 0:
            self.status_info.setText("No mesh data to export!")
            return
            
        path, _ = QFileDialog.getSaveFileName(self, "Save OBJ", "blob_export.obj", "OBJ Files (*.obj)")
        if path:
            md = self.mesh_item.meshData()
            v = md.vertexes()
            f = md.faces()
            with open(path, 'w') as file:
                file.write("# Generated by 3D Blob Visualizer\n")
                for vert in v:
                    file.write(f"v {vert[0]} {vert[1]} {vert[2]}\n")
                for face in f:
                    file.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
            self.status_info.setText(f"Exported mesh to {path}")

    def _screenshot(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Screenshot", "blob_screenshot.png", "PNG Files (*.png)")
        if path:
            img = self.view.grabFramebuffer()
            img.save(path)
            self.status_info.setText(f"Saved screenshot to {path}")

    def _on_pause(self, checked):
        self.paused = checked
        self.pause_btn.setText("▶ Resume" if checked else "⏸ Pause")

    def keyPressEvent(self, ev: QKeyEvent):
        k = ev.key()
        if k == Qt.Key_Escape: self.close()
        elif k == Qt.Key_Space: self.pause_btn.toggle()
        elif k == Qt.Key_R: self._reset()
        else: super().keyPressEvent(ev)

    def closeEvent(self, event):
        self.worker_thread.quit()
        self.worker_thread.wait()
        event.accept()

# ---------- Entry point -------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    f = QFont()
    f.setFamilies(["Segoe UI", "SF Pro Text", "Helvetica Neue", "Arial"])
    f.setPointSize(9)
    app.setFont(f)
    pg.setConfigOptions(antialias=True)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()