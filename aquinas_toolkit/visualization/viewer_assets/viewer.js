import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// ─── Constants ────────────────────────────────────────────────────────────────

const SCALE = 3; // normalized coord → world unit multiplier
const SIDE_Z = { UP: 0.24 * SCALE, DO: -0.24 * SCALE };

const COLOR = {
  deckOld: 0xd9d0c0,
  deckNew: 0xbfcfc8,
  deckEdge: 0x2a2420,
  pier: 0x9a9088,
  accRed: 0xc53030,
  strBlue: 0x2b6cb0,
  highAmber: 0xd97706,
  selected: 0xf5a623,
};

const CAM_DEFAULT = { x: 3 * SCALE, y: 2.8 * SCALE, z: 10 * SCALE };
const CAM_TARGET  = { x: 1.0 * SCALE, y: 0, z: 0 };

// ─── State ────────────────────────────────────────────────────────────────────

const state = {
  manifest: null,
  geometry: null,
  sensorLayout: [],
  metrics: [],
  trends: [],
  eventGroups: [],
  correlations: [],
  metricLookup: new Map(),
  trendLookup: new Map(),
  correlationLookup: new Map(),
  eventLookup: new Map(),
  waveformCache: new Map(),
  selectedDataset: null,
  selectedMetric: "mean_range",
  family: "ALL",
  viewMode: "exploded",
  compareMode: "single",
  showCorrelations: false,
  selectedSensorId: null,
  isolatedSensorId: null,
  selectedPreviewEventId: null,
  filters: {
    deck: new Set(),
    span: new Set(),
    side: new Set(),
    section: new Set(),
    axis_or_fibre: new Set(),
  },
};

// ─── Three.js context ─────────────────────────────────────────────────────────

let renderer, scene, camera, controls, raycaster, pointer;
const deckGroups = {}; // deck → THREE.Group (positioned at deck Z center)
const sensorObjs = new Map(); // sensorId → { group: THREE.Group, meshes: THREE.Mesh[] }
const pickTargets = []; // flat array of meshes tested by raycaster
let hoveredId = null;
let tweenId = null;

// ─── DOM refs ─────────────────────────────────────────────────────────────────

const el = {
  datasetSelect: document.getElementById("dataset-select"),
  metricSelect:  document.getElementById("metric-select"),
  familyToggle:  document.getElementById("family-toggle"),
  viewToggle:    document.getElementById("view-toggle"),
  compareMode:   document.getElementById("compare-mode"),
  corrToggle:    document.getElementById("correlation-toggle"),
  filterGroups:  document.getElementById("filter-groups"),
  resetFilters:  document.getElementById("reset-filters"),
  resetCamera:   document.getElementById("reset-camera"),
  stageTitle:    document.getElementById("stage-title"),
  stageSub:      document.getElementById("stage-subtitle"),
  statusStrip:   document.getElementById("status-strip"),
  sceneWrap:     document.getElementById("scene-wrap"),
  canvas:        document.getElementById("bridge-canvas"),
  tooltip:       document.getElementById("tooltip"),
  inspector:     document.getElementById("inspector"),
  datasetStrip:  document.getElementById("dataset-strip"),
};

// ─── Boot ─────────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  initialize().catch((err) => {
    el.inspector.innerHTML = `
      <div class="card">
        <h3>Viewer failed to load</h3>
        <p class="card-note">${err.message}</p>
      </div>`;
  });
});

async function initialize() {
  state.manifest = await fetchJson("./manifest.json");
  const files = state.manifest.files;

  const [geometry, sensorLayout, metrics, trends, eventGroups, correlations] =
    await Promise.all([
      fetchJson(files.bridge_geometry),
      fetchJson(files.sensor_layout),
      fetchJson(files.sensor_metrics),
      fetchJson(files.sensor_trends),
      fetchJson(files.event_groups),
      fetchJson(files.correlations),
    ]);

  state.geometry     = geometry;
  state.sensorLayout = sensorLayout;
  state.metrics      = metrics;
  state.trends       = trends;
  state.eventGroups  = eventGroups;
  state.correlations = correlations;
  state.selectedDataset = state.manifest.default_dataset;

  buildLookups();
  buildControls();
  bindStaticEvents();
  initThree();
  buildBridgeScene();
  render();
}

// ─── Lookups ──────────────────────────────────────────────────────────────────

function buildLookups() {
  state.metricLookup = new Map(
    state.metrics.map((r) => [metricKey(r.dataset, r.metric_id, r.sensor_id), r])
  );
  state.trendLookup = new Map(
    state.trends.map((r) => [`${r.sensor_id}|${r.metric_id}`, r])
  );
  state.correlationLookup = new Map();
  for (const r of state.correlations) {
    const k = `${r.sensor_id}|${r.metric_id}`;
    const cur = state.correlationLookup.get(k) || [];
    cur.push(r);
    state.correlationLookup.set(k, cur);
  }
  state.eventLookup = new Map();
  for (const r of state.eventGroups) {
    const k = `${r.dataset}|${r.deck}`;
    const cur = state.eventLookup.get(k) || [];
    cur.push(r);
    state.eventLookup.set(k, cur);
  }
}

// ─── Controls ─────────────────────────────────────────────────────────────────

function buildControls() {
  el.datasetSelect.innerHTML = state.manifest.available_datasets
    .map((d) => `<option value="${d.dataset}">${d.label} · ${d.dataset}</option>`)
    .join("");
  el.datasetSelect.value = state.selectedDataset;

  el.metricSelect.innerHTML = state.manifest.metric_catalog
    .map((m) => `<option value="${m.metric_id}">${m.label}</option>`)
    .join("");
  el.metricSelect.value = state.selectedMetric;

  renderSegmented(el.familyToggle, ["ALL", "ACC", "STR"], state.family, (v) => {
    state.family = v;
    reconcileSelection();
    render();
  });
  renderSegmented(el.viewToggle, ["exploded", "compact"], state.viewMode, (v) => {
    state.viewMode = v;
    tweenDeckPositions();
    render();
  });

  buildFilterControls();
  buildDatasetStrip();
}

function buildFilterControls() {
  const activeFamily = state.family;
  const axisValues = uniqueValues(
    state.sensorLayout
      .filter((s) => activeFamily === "ALL" || s.measurement_family === activeFamily)
      .map((s) => s.axis_or_fibre)
  );
  const groups = [
    ["deck",          uniqueValues(state.sensorLayout.map((s) => s.deck))],
    ["span",          uniqueValues(state.sensorLayout.map((s) => s.span))],
    ["side",          uniqueValues(state.sensorLayout.map((s) => s.side))],
    ["section",       uniqueValues(state.sensorLayout.map((s) => s.section))],
    ["axis_or_fibre", axisValues],
  ];

  el.filterGroups.innerHTML = groups
    .map(([name, values]) => {
      const selected = state.filters[name];
      const options = values
        .map((v) => {
          const checked = selected.size === 0 || selected.has(v) ? "checked" : "";
          return `<label>
            <input type="checkbox" data-filter-group="${name}" value="${v}" ${checked} />
            <span>${v}</span>
          </label>`;
        })
        .join("");
      const title = name === "axis_or_fibre" ? "Axis / fibre" : capitalize(name);
      return `<div class="filter-group">
        <h3>${title}</h3>
        <div class="check-grid">${options}</div>
      </div>`;
    })
    .join("");
}

function buildDatasetStrip() {
  el.datasetStrip.innerHTML = state.manifest.available_datasets
    .map(
      (d) => `<button class="dataset-button ${d.dataset === state.selectedDataset ? "active" : ""}"
               data-dataset-button="${d.dataset}" type="button">
        <strong>${d.label}</strong>
        <span>${d.dataset}</span>
      </button>`
    )
    .join("");
}

function bindStaticEvents() {
  el.datasetSelect.addEventListener("change", (e) => {
    state.selectedDataset = e.target.value;
    render();
  });
  el.metricSelect.addEventListener("change", (e) => {
    state.selectedMetric = e.target.value;
    render();
  });
  el.compareMode.addEventListener("change", (e) => {
    state.compareMode = e.target.value;
    render();
  });
  el.corrToggle.addEventListener("change", (e) => {
    state.showCorrelations = e.target.checked;
    render();
  });
  el.resetFilters.addEventListener("click", () => {
    for (const k of Object.keys(state.filters)) state.filters[k] = new Set();
    state.isolatedSensorId = null;
    buildFilterControls();
    render();
  });
  el.resetCamera.addEventListener("click", resetCamera);
  el.filterGroups.addEventListener("change", (e) => {
    const t = e.target;
    if (!(t instanceof HTMLInputElement) || !t.dataset.filterGroup) return;
    const grp = t.dataset.filterGroup;
    const checked = new Set(
      Array.from(el.filterGroups.querySelectorAll(`input[data-filter-group="${grp}"]:checked`))
        .map((i) => i.value)
    );
    const all = new Set(
      Array.from(el.filterGroups.querySelectorAll(`input[data-filter-group="${grp}"]`))
        .map((i) => i.value)
    );
    state.filters[grp] = checked.size === all.size ? new Set() : checked;
    reconcileSelection();
    render();
  });
  el.datasetStrip.addEventListener("click", (e) => {
    const t = e.target.closest("[data-dataset-button]");
    if (!t) return;
    state.selectedDataset = t.dataset.datasetButton;
    el.datasetSelect.value = state.selectedDataset;
    render();
  });
}

// ─── Render orchestrator ──────────────────────────────────────────────────────

function render() {
  buildFilterControls();
  buildDatasetStrip();
  updateHeader();
  updateSensorGlyphs();
  renderInspector();
}

function updateHeader() {
  const dataset = state.manifest.available_datasets.find(
    (d) => d.dataset === state.selectedDataset
  );
  const visible = getVisibleSensors();
  el.stageTitle.textContent = `${dataset ? dataset.label : state.selectedDataset} · ${labelForMetric(state.selectedMetric)}`;
  el.stageSub.textContent = "3D analytical schematic — drag to orbit, scroll to zoom.";
  el.statusStrip.innerHTML = [
    `${visible.length} sensors`,
    `${state.family} family`,
    capitalize(state.viewMode),
  ]
    .map((t) => `<span class="status-pill">${t}</span>`)
    .join("");
}

// ─── Three.js scene setup ─────────────────────────────────────────────────────

function initThree() {
  renderer = new THREE.WebGLRenderer({ canvas: el.canvas, antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setClearColor(0xf7f5f0, 1);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0xf7f5f0, 0.018);

  camera = new THREE.PerspectiveCamera(42, 1, 0.05, 300);
  camera.position.set(CAM_DEFAULT.x, CAM_DEFAULT.y, CAM_DEFAULT.z);

  controls = new OrbitControls(camera, el.canvas);
  controls.target.set(CAM_TARGET.x, CAM_TARGET.y, CAM_TARGET.z);
  controls.enableDamping = true;
  controls.dampingFactor = 0.07;
  controls.minPolarAngle = 0.05;
  controls.maxPolarAngle = Math.PI * 0.48;
  controls.minDistance = 2;
  controls.maxDistance = 60;
  controls.update();

  // Lighting
  const ambient = new THREE.AmbientLight(0xffffff, 0.65);
  scene.add(ambient);

  const sun = new THREE.DirectionalLight(0xfff4e0, 1.1);
  sun.position.set(8, 18, 12);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.camera.near = 0.5;
  sun.shadow.camera.far = 80;
  sun.shadow.camera.left = -20;
  sun.shadow.camera.right = 20;
  sun.shadow.camera.top = 10;
  sun.shadow.camera.bottom = -10;
  sun.shadow.bias = -0.001;
  scene.add(sun);

  const fill = new THREE.DirectionalLight(0xd0e8ff, 0.35);
  fill.position.set(-6, 5, -10);
  scene.add(fill);

  // Ground plane (subtle shadow receiver)
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(80, 80),
    new THREE.ShadowMaterial({ opacity: 0.06 })
  );
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = -1.2 * SCALE;
  ground.receiveShadow = true;
  scene.add(ground);

  raycaster = new THREE.Raycaster();
  raycaster.params.Line = { threshold: 0.1 };
  pointer = new THREE.Vector2();

  el.canvas.addEventListener("pointermove", onPointerMove);
  el.canvas.addEventListener("click", onCanvasClick);
  el.canvas.addEventListener("dblclick", onCanvasDoubleClick);

  const ro = new ResizeObserver(onResize);
  ro.observe(el.sceneWrap);
  onResize();

  animate();
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

function onResize() {
  const w = el.sceneWrap.clientWidth;
  const h = el.sceneWrap.clientHeight || 480;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

function resetCamera() {
  camera.position.set(CAM_DEFAULT.x, CAM_DEFAULT.y, CAM_DEFAULT.z);
  controls.target.set(CAM_TARGET.x, CAM_TARGET.y, CAM_TARGET.z);
  controls.update();
}

// ─── Bridge geometry ──────────────────────────────────────────────────────────

function buildBridgeScene() {
  buildDecks();
  buildPiers();
}

function buildDecks() {
  const viewModes = state.geometry.view_modes;

  for (const deckData of state.geometry.deck_meshes) {
    const group = new THREE.Group();
    deckGroups[deckData.deck] = group;
    scene.add(group);

    const deckColor = deckData.deck === "OLD" ? COLOR.deckOld : COLOR.deckNew;
    const zCenter = viewModes[state.viewMode].deck_centers[deckData.deck];
    group.position.z = zCenter * SCALE;

    for (const seg of deckData.segments) {
      const len   = (seg.x_end - seg.x_start) * SCALE;
      const cx    = ((seg.x_start + seg.x_end) / 2) * SCALE;
      const w     = seg.width * SCALE;   // transverse (local Z within group)
      const d     = seg.depth * SCALE;   // vertical (Y)

      // Main girder — hollow box-girder cross-section via CSG-like approach:
      // outer box + two inner cutout boxes (flanges) using MeshStandardMaterial
      const outerGeo = new THREE.BoxGeometry(len, d, w);
      const mat = new THREE.MeshStandardMaterial({
        color: deckColor,
        roughness: 0.78,
        metalness: 0.0,
      });
      const mesh = new THREE.Mesh(outerGeo, mat);
      mesh.position.set(cx, 0, 0);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      group.add(mesh);

      // Sharp edge overlay
      const edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(outerGeo, 20),
        new THREE.LineBasicMaterial({ color: COLOR.deckEdge, linewidth: 1 })
      );
      edges.position.set(cx, 0, 0);
      group.add(edges);

      // Top flange highlight strip
      const flangeGeo = new THREE.BoxGeometry(len, d * 0.09, w);
      const flangeMat = new THREE.MeshStandardMaterial({
        color: deckData.deck === "OLD" ? 0xe8dece : 0xd0dedc,
        roughness: 0.6,
      });
      const flange = new THREE.Mesh(flangeGeo, flangeMat);
      flange.position.set(cx, d * 0.455, 0);
      flange.receiveShadow = true;
      group.add(flange);
    }

    // Deck label sprite
    addDeckLabel(group, deckData.deck, viewModes[state.viewMode].deck_centers[deckData.deck]);
  }
}

function addDeckLabel(group, deckName, _zCenter) {
  // Floating text via canvas sprite
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 64;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "rgba(0,0,0,0)";
  ctx.fillRect(0, 0, 256, 64);
  ctx.font = "bold 32px 'Barlow Condensed', sans-serif";
  ctx.fillStyle = deckName === "OLD" ? "#8b7355" : "#4a7a6e";
  ctx.textAlign = "left";
  ctx.fillText(`${deckName} DECK`, 12, 44);

  const texture = new THREE.CanvasTexture(canvas);
  const mat = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(2.5, 0.65, 1);
  sprite.position.set(0.2 * SCALE, 0.6 * SCALE, 0);
  group.add(sprite);
}

function buildPiers() {
  for (const pier of state.geometry.pier_anchors) {
    const wx = pier.x * SCALE;
    // Horizontal cross-beam spanning both decks
    const beamH = 0.12 * SCALE;
    const beamW = 0.08 * SCALE;
    const beamSpan = 4.5 * SCALE; // spans z from OLD to NEW
    const beamGeo = new THREE.BoxGeometry(beamW, beamH, beamSpan);
    const beamMat = new THREE.MeshStandardMaterial({ color: COLOR.pier, roughness: 0.85 });
    const beam = new THREE.Mesh(beamGeo, beamMat);
    beam.position.set(wx, -0.28 * SCALE, 0);
    beam.castShadow = true;
    scene.add(beam);

    // Vertical support column (stub)
    const colGeo = new THREE.CylinderGeometry(0.06 * SCALE, 0.09 * SCALE, 0.45 * SCALE, 6);
    const col = new THREE.Mesh(colGeo, beamMat);
    col.position.set(wx, -0.58 * SCALE, 0);
    col.castShadow = true;
    scene.add(col);

    // Triangular footing (inverted pyramid)
    const footGeo = new THREE.ConeGeometry(0.18 * SCALE, 0.24 * SCALE, 3);
    const foot = new THREE.Mesh(footGeo, beamMat);
    foot.rotation.x = Math.PI;
    foot.position.set(wx, -0.95 * SCALE, 0);
    foot.castShadow = true;
    scene.add(foot);
  }
}

// ─── View mode tween ──────────────────────────────────────────────────────────

function tweenDeckPositions() {
  const viewModes = state.geometry.view_modes;
  const targets = {};
  for (const deck of ["OLD", "NEW"]) {
    targets[deck] = viewModes[state.viewMode].deck_centers[deck] * SCALE;
  }

  if (tweenId) cancelAnimationFrame(tweenId);

  const duration = 420;
  const start = performance.now();
  const fromZ = { OLD: deckGroups.OLD?.position.z ?? targets.OLD, NEW: deckGroups.NEW?.position.z ?? targets.NEW };

  function step(now) {
    const t = Math.min((now - start) / duration, 1);
    const ease = 1 - Math.pow(1 - t, 3); // cubic ease-out
    for (const deck of ["OLD", "NEW"]) {
      if (deckGroups[deck]) {
        deckGroups[deck].position.z = fromZ[deck] + (targets[deck] - fromZ[deck]) * ease;
      }
    }
    // Also tween sensor glyph positions
    for (const [sensorId, obj] of sensorObjs) {
      const sensor = state.sensorLayout.find((s) => s.sensor_id === sensorId);
      if (!sensor) continue;
      const targetZ = viewModeZ(sensor);
      obj.group.position.z = fromZ[sensor.deck] + (targets[sensor.deck] - fromZ[sensor.deck]) * ease + SIDE_Z[sensor.side];
      // Actually sensor glyph is in scene (not in deck group), so compute absolute Z
      obj.group.position.z = fromZ[sensor.deck] + (targets[sensor.deck] - fromZ[sensor.deck]) * ease + SIDE_Z[sensor.side];
    }
    if (t < 1) tweenId = requestAnimationFrame(step);
  }

  tweenId = requestAnimationFrame(step);
}

// ─── Sensor glyphs ────────────────────────────────────────────────────────────

function updateSensorGlyphs() {
  const visible = new Set(getVisibleSensors().map((s) => s.sensor_id));
  const selected = state.selectedSensorId;

  // Remove glyphs for sensors that are no longer visible
  for (const [id, obj] of sensorObjs) {
    if (!visible.has(id)) {
      scene.remove(obj.group);
      sensorObjs.delete(id);
      const idx = pickTargets.indexOf(obj.pickMesh);
      if (idx !== -1) pickTargets.splice(idx, 1);
    }
  }

  // Add or update glyphs
  for (const sensor of getVisibleSensors()) {
    const metric = getMetric(sensor.sensor_id);
    const isSelected = sensor.sensor_id === selected;
    const isHigh = metric?.status_band === "high";
    const isLow  = metric?.status_band === "low";

    if (sensorObjs.has(sensor.sensor_id)) {
      // Update color/scale
      const obj = sensorObjs.get(sensor.sensor_id);
      const col = glyphColor(sensor, isHigh, isSelected);
      for (const m of obj.meshes) {
        m.material.color.setHex(col);
        m.material.emissive?.setHex(isSelected ? 0x331100 : 0x000000);
        m.material.opacity = isLow ? 0.38 : 1.0;
      }
      obj.group.scale.setScalar(isSelected ? 1.45 : (sensor.sensor_id === hoveredId ? 1.3 : 1.0));
    } else {
      // Create new glyph
      const group = buildGlyph(sensor, metric, isSelected);
      group.position.set(
        sensor.x * SCALE,
        sensor.y * SCALE,
        viewModeZ(sensor)
      );
      scene.add(group);

      // Invisible pick sphere for raycasting
      const pickGeo = new THREE.SphereGeometry(0.18 * SCALE, 6, 6);
      const pickMat = new THREE.MeshBasicMaterial({ visible: false });
      const pickMesh = new THREE.Mesh(pickGeo, pickMat);
      pickMesh.userData.sensorId = sensor.sensor_id;
      group.add(pickMesh);
      pickTargets.push(pickMesh);

      sensorObjs.set(sensor.sensor_id, { group, meshes: group.userData.coloredMeshes, pickMesh });
    }
  }
}

function viewModeZ(sensor) {
  const deck = sensor.deck;
  const deckZ = state.geometry.view_modes[state.viewMode].deck_centers[deck] * SCALE;
  return deckZ + SIDE_Z[sensor.side];
}

function glyphColor(sensor, isHigh, isSelected) {
  if (isSelected) return COLOR.selected;
  if (isHigh)     return COLOR.highAmber;
  return sensor.measurement_family === "ACC" ? COLOR.accRed : COLOR.strBlue;
}

function buildGlyph(sensor, metric, isSelected) {
  const group = new THREE.Group();
  const isHigh = metric?.status_band === "high";
  const col = glyphColor(sensor, isHigh, isSelected);
  const meshes = [];

  const mat = () =>
    new THREE.MeshStandardMaterial({
      color: col,
      roughness: 0.4,
      metalness: 0.1,
      emissive: isSelected ? new THREE.Color(0x331100) : new THREE.Color(0x000000),
      transparent: true,
      opacity: metric?.status_band === "low" ? 0.38 : 1.0,
    });

  const sz = 0.13 * SCALE; // base glyph size

  if (sensor.glyph_type === "vertical-arrow") {
    // Upward arrow: shaft + cone
    const shaft = new THREE.Mesh(new THREE.CylinderGeometry(sz * 0.22, sz * 0.22, sz * 1.6, 8), mat());
    shaft.position.y = sz * 0.5;
    const head = new THREE.Mesh(new THREE.ConeGeometry(sz * 0.5, sz * 0.85, 8), mat());
    head.position.y = sz * 1.6;
    meshes.push(shaft, head);
    group.add(shaft, head);

  } else if (sensor.glyph_type === "transverse-arrow") {
    // Sideways arrow in Z direction
    const shaft = new THREE.Mesh(new THREE.CylinderGeometry(sz * 0.22, sz * 0.22, sz * 1.6, 8), mat());
    shaft.rotation.x = Math.PI / 2;
    shaft.position.z = sz * 0.5;
    const head = new THREE.Mesh(new THREE.ConeGeometry(sz * 0.5, sz * 0.85, 8), mat());
    head.rotation.x = Math.PI / 2;
    head.position.z = sz * 1.55;
    meshes.push(shaft, head);
    group.add(shaft, head);

  } else {
    // Strain gauge — double-headed bone shape (two cones pointing out from center)
    const body = new THREE.Mesh(new THREE.CylinderGeometry(sz * 0.18, sz * 0.18, sz * 1.4, 8), mat());
    body.rotation.z = Math.PI / 2; // horizontal
    const headL = new THREE.Mesh(new THREE.ConeGeometry(sz * 0.48, sz * 0.7, 8), mat());
    headL.rotation.z = Math.PI / 2;
    headL.position.x = -sz * 1.1;
    const headR = new THREE.Mesh(new THREE.ConeGeometry(sz * 0.48, sz * 0.7, 8), mat());
    headR.rotation.z = -Math.PI / 2;
    headR.position.x = sz * 1.1;
    meshes.push(body, headL, headR);
    group.add(body, headL, headR);
  }

  group.userData.coloredMeshes = meshes;
  for (const m of meshes) {
    m.castShadow = true;
  }
  return group;
}

// ─── Raycasting ───────────────────────────────────────────────────────────────

function getCanvasPointer(event) {
  const rect = el.canvas.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / rect.width) * 2 - 1,
    y: -((event.clientY - rect.top) / rect.height) * 2 + 1,
  };
}

function raycastSensor(event) {
  const p = getCanvasPointer(event);
  pointer.set(p.x, p.y);
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObjects(pickTargets, false);
  return hits.length ? hits[0].object.userData.sensorId : null;
}

function onPointerMove(event) {
  const sensorId = raycastSensor(event);
  if (sensorId !== hoveredId) {
    // Update previous hovered
    if (hoveredId && sensorObjs.has(hoveredId)) {
      const obj = sensorObjs.get(hoveredId);
      const isSelected = hoveredId === state.selectedSensorId;
      obj.group.scale.setScalar(isSelected ? 1.45 : 1.0);
    }
    hoveredId = sensorId;
    // Update new hovered
    if (hoveredId && sensorObjs.has(hoveredId) && hoveredId !== state.selectedSensorId) {
      sensorObjs.get(hoveredId).group.scale.setScalar(1.3);
    }
  }

  if (sensorId) {
    const sensor = state.sensorLayout.find((s) => s.sensor_id === sensorId);
    const metric = getMetric(sensorId);
    el.tooltip.innerHTML = `
      <strong>${sensorId}</strong><br>
      ${sensor.deck} · ${sensor.span}_${sensor.section} · ${sensor.side}<br>
      ${labelForMetric(state.selectedMetric)}: ${formatMetric(metric)}`;
    el.tooltip.classList.remove("hidden");
    el.tooltip.style.left = `${event.offsetX}px`;
    el.tooltip.style.top  = `${event.offsetY}px`;
  } else {
    el.tooltip.classList.add("hidden");
  }
}

function onCanvasClick(event) {
  const sensorId = raycastSensor(event);
  if (!sensorId) return;
  state.selectedSensorId = sensorId;
  state.selectedPreviewEventId = null;
  ensureDefaultPreview().finally(() => render());
}

function onCanvasDoubleClick(event) {
  const sensorId = raycastSensor(event);
  if (!sensorId) return;
  state.isolatedSensorId = state.isolatedSensorId === sensorId ? null : sensorId;
  state.selectedSensorId = sensorId;
  render();
}

// ─── Inspector ────────────────────────────────────────────────────────────────

function renderInspector() {
  const sensor = state.sensorLayout.find((s) => s.sensor_id === state.selectedSensorId);
  if (!sensor) {
    el.inspector.innerHTML = `
      <div class="card">
        <h3>No sensor selected</h3>
        <p class="card-note">Click any marker in the 3D view to pin a sensor and inspect
        its metric value, trend, homologous comparison, correlations, and waveform preview.</p>
      </div>`;
    return;
  }

  const metric          = getMetric(sensor.sensor_id);
  const homologous      = sensor.homologous_sensor_id
    ? state.sensorLayout.find((s) => s.sensor_id === sensor.homologous_sensor_id)
    : null;
  const homologousMetric = homologous ? getMetric(homologous.sensor_id) : null;
  const relatedSensors   = getRelatedSensors(sensor);
  const correlations     = (
    state.correlationLookup.get(`${sensor.sensor_id}|${state.selectedMetric}`) || []
  ).slice(0, 3);
  const trend = state.trendLookup.get(`${sensor.sensor_id}|${state.selectedMetric}`);
  const waveformCard = renderWaveformCard(sensor);

  el.inspector.innerHTML = `
    <div class="card">
      <h3>${sensor.sensor_id}</h3>
      <div class="metric-value">${formatMetric(metric)}</div>
      <p class="card-note">${labelForMetric(state.selectedMetric)} · ${metric?.unit ?? "n/a"}</p>
    </div>

    <div class="card">
      <h3>Location</h3>
      <div class="meta-grid">
        ${metaRow("Deck", sensor.deck)}
        ${metaRow("Span", sensor.span)}
        ${metaRow("Side", sensor.side)}
        ${metaRow("Section", `${sensor.span}_${sensor.section}`)}
        ${metaRow("Family", sensor.measurement_family)}
        ${metaRow("Axis / fibre", sensor.axis_or_fibre)}
      </div>
    </div>

    <div class="card">
      <h3>Homologous comparison</h3>
      ${
        homologous
          ? `<div class="meta-grid">
              ${metaRow("Paired sensor", homologous.sensor_id)}
              ${metaRow("This value", formatMetric(metric))}
              ${metaRow("Paired value", formatMetric(homologousMetric))}
            </div>`
          : `<p class="empty-state">No homologous sensor in exported layout.</p>`
      }
    </div>

    <div class="card">
      <h3>Trend across datasets</h3>
      ${
        trend
          ? `${renderSparkline(trend.points)}
             <p class="card-note">${trend.points
               .map((p) => `${p.dataset_label}: ${formatNumeric(p.value)}`)
               .join(" · ")}</p>`
          : `<p class="empty-state">No trend points available for this metric.</p>`
      }
    </div>

    <div class="card">
      <h3>Family-aware comparisons</h3>
      ${
        relatedSensors.length
          ? `<ul class="mini-list">${relatedSensors
              .map((e) => `<li>${e.sensor_id} · ${formatMetric(getMetric(e.sensor_id))}</li>`)
              .join("")}</ul>`
          : `<p class="empty-state">No related sensors under active filters.</p>`
      }
    </div>

    <div class="card">
      <h3>Top correlations</h3>
      ${
        correlations.length
          ? `<ul class="mini-list">${correlations
              .map(
                (r) => `<li>${r.target_sensor_id} · r=${formatNumeric(r.correlation)}${
                  r.cross_deck_homologous ? " · homologous" : ""
                }</li>`
              )
              .join("")}</ul>`
          : `<p class="empty-state">Correlations based on proxy metric trends.</p>`
      }
    </div>

    ${waveformCard}
  `;

  const previewSelect = el.inspector.querySelector("[data-waveform-select]");
  if (previewSelect) {
    previewSelect.addEventListener("change", async (e) => {
      const id = e.target.value;
      if (!id) { state.selectedPreviewEventId = null; renderInspector(); return; }
      state.selectedPreviewEventId = id;
      await ensureWaveformLoaded(id);
      renderInspector();
    });
  }
}

function renderWaveformCard(sensor) {
  const events = (state.eventLookup.get(`${state.selectedDataset}|${sensor.deck}`) || [])
    .filter((ev) => ev.sensor_ids.includes(sensor.sensor_id));
  if (!events.length) {
    return `<div class="card">
      <h3>Deck-scoped event detail</h3>
      <p class="empty-state">No event groups exported for this deck and dataset.</p>
    </div>`;
  }

  const selectable = events.filter((ev) => ev.waveform_preview_path);
  const activeId =
    selectable.find((ev) => ev.event_group_id === state.selectedPreviewEventId)?.event_group_id ||
    selectable[0]?.event_group_id || "";
  const waveformData = activeId ? state.waveformCache.get(activeId) : null;
  const trace = waveformData?.traces?.find((t) => t.sensor_id === sensor.sensor_id);

  return `<div class="card">
    <h3>Deck-scoped event detail</h3>
    <p class="card-note">Events are keyed by dataset + deck + time window.</p>
    <div class="control">
      <label for="waveform-select">Preview event</label>
      <select id="waveform-select" data-waveform-select>
        ${selectable
          .map(
            (ev) => `<option value="${ev.event_group_id}" ${ev.event_group_id === activeId ? "selected" : ""}>
              ${ev.start_time_utc} · ${ev.sensor_count} sensors
            </option>`
          )
          .join("")}
      </select>
    </div>
    ${
      !selectable.length
        ? `<p class="empty-state">Waveform previews not included in this bundle.</p>`
        : trace
        ? renderWaveform(trace)
        : `<p class="empty-state">Select an event to inspect this sensor's waveform.</p>`
    }
  </div>`;
}

async function ensureWaveformLoaded(eventGroupId) {
  if (state.waveformCache.has(eventGroupId)) return;
  const ev = state.eventGroups.find((e) => e.event_group_id === eventGroupId);
  if (!ev?.waveform_preview_path) return;
  const payload = await fetchJson(ev.waveform_preview_path);
  state.waveformCache.set(eventGroupId, payload);
}

async function ensureDefaultPreview() {
  const sensor = state.sensorLayout.find((s) => s.sensor_id === state.selectedSensorId);
  if (!sensor) return;
  const ev = (state.eventLookup.get(`${state.selectedDataset}|${sensor.deck}`) || []).find(
    (e) => e.waveform_preview_path && e.sensor_ids.includes(sensor.sensor_id)
  );
  if (!ev) return;
  state.selectedPreviewEventId = ev.event_group_id;
  await ensureWaveformLoaded(ev.event_group_id);
}

// ─── Sparkline / Waveform (SVG, unchanged from original) ─────────────────────

function renderSparkline(points) {
  const valid = points.filter((p) => p.value !== null && p.value !== undefined);
  if (!valid.length) return "";
  const W = 250, H = 100;
  const min = Math.min(...valid.map((p) => p.value));
  const max = Math.max(...valid.map((p) => p.value));
  const range = max - min || 1;
  const path = valid
    .map((p, i) => {
      const x = 14 + (i * (W - 28)) / Math.max(valid.length - 1, 1);
      const y = H - 14 - ((p.value - min) / range) * (H - 28);
      return `${i === 0 ? "M" : "L"} ${x} ${y}`;
    })
    .join(" ");
  const area = `M 14 ${H - 14} ` + valid
    .map((p, i) => {
      const x = 14 + (i * (W - 28)) / Math.max(valid.length - 1, 1);
      const y = H - 14 - ((p.value - min) / range) * (H - 28);
      return `L ${x} ${y}`;
    })
    .join(" ") + ` L ${14 + ((valid.length - 1) * (W - 28)) / Math.max(valid.length - 1, 1)} ${H - 14} Z`;
  const dots = valid
    .map((p, i) => {
      const x = 14 + (i * (W - 28)) / Math.max(valid.length - 1, 1);
      const y = H - 14 - ((p.value - min) / range) * (H - 28);
      return `<circle cx="${x}" cy="${y}" r="3.5" fill="var(--accent)"/>`;
    })
    .join("");
  return `<svg class="sparkline" viewBox="0 0 ${W} ${H}">
    <path d="${area}" fill="var(--accent)" opacity="0.08"/>
    <path d="${path}" fill="none" stroke="var(--accent)" stroke-width="2.2"/>
    ${dots}
  </svg>`;
}

function renderWaveform(trace) {
  const values = trace.values || [];
  if (!values.length) return `<p class="empty-state">No waveform samples.</p>`;
  const W = 250, H = 120;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const path = values
    .map((v, i) => {
      const x = 12 + (i * (W - 24)) / Math.max(values.length - 1, 1);
      const y = H - 14 - ((v - min) / range) * (H - 28);
      return `${i === 0 ? "M" : "L"} ${x} ${y}`;
    })
    .join(" ");
  return `<svg class="waveform" viewBox="0 0 ${W} ${H}">
    <line x1="12" y1="${H / 2}" x2="${W - 12}" y2="${H / 2}" stroke="var(--border)" stroke-width="1"/>
    <path d="${path}" fill="none" stroke="var(--acc)" stroke-width="1.8"/>
  </svg>`;
}

// ─── Visibility / filtering (unchanged logic) ─────────────────────────────────

function getVisibleSensors() {
  return state.sensorLayout.filter(isVisible);
}

function isVisible(sensor) {
  if (state.family !== "ALL" && sensor.measurement_family !== state.family) return false;
  if (state.isolatedSensorId) {
    const iso = state.sensorLayout.find((s) => s.sensor_id === state.isolatedSensorId);
    if (iso) {
      const sameNeighbourhood =
        sensor.deck === iso.deck &&
        sensor.span === iso.span &&
        sensor.section === iso.section;
      const isPair =
        sensor.sensor_id === iso.homologous_sensor_id ||
        sensor.sensor_id === iso.sensor_id;
      if (!sameNeighbourhood && !isPair) return false;
    }
  }
  return (
    passesFilter("deck",          sensor.deck) &&
    passesFilter("span",          sensor.span) &&
    passesFilter("side",          sensor.side) &&
    passesFilter("section",       sensor.section) &&
    passesFilter("axis_or_fibre", sensor.axis_or_fibre)
  );
}

function passesFilter(group, value) {
  const sel = state.filters[group];
  return sel.size === 0 || sel.has(value);
}

function reconcileSelection() {
  if (!state.selectedSensorId) return;
  const s = state.sensorLayout.find((e) => e.sensor_id === state.selectedSensorId);
  if (!s || !isVisible(s)) state.selectedSensorId = null;
}

function getRelatedSensors(sensor) {
  const fam = sensor.measurement_family;
  return state.sensorLayout
    .filter((e) => {
      if (e.sensor_id === sensor.sensor_id) return false;
      if (state.family !== "ALL" && e.measurement_family !== state.family) return false;
      if (fam !== e.measurement_family) return false;
      if (fam === "ACC")
        return e.deck === sensor.deck && e.span === sensor.span && e.section === sensor.section;
      return e.deck === sensor.deck && e.span === sensor.span;
    })
    .slice(0, 4);
}

// ─── Utilities ────────────────────────────────────────────────────────────────

function getMetric(sensorId) {
  return state.metricLookup.get(metricKey(state.selectedDataset, state.selectedMetric, sensorId));
}

function renderSegmented(container, values, active, onChange) {
  container.innerHTML = values
    .map(
      (v) => `<button type="button" class="${v === active ? "active" : ""}" data-segment-value="${v}">${v}</button>`
    )
    .join("");
  container.addEventListener("click", (e) => {
    const t = e.target.closest("[data-segment-value]");
    if (t) onChange(t.dataset.segmentValue);
  });
}

function fetchJson(path) {
  return fetch(path).then(async (r) => {
    if (!r.ok) throw new Error(`Failed to load ${path}: ${r.status}`);
    return r.json();
  });
}

function uniqueValues(values) { return [...new Set(values)].sort(); }

function labelForMetric(id) {
  const m = state.manifest.metric_catalog.find((e) => e.metric_id === id);
  return m ? m.label : id;
}

function formatMetric(metric) {
  if (!metric) return "n/a";
  return `${formatNumeric(metric.value)} ${metric.unit}`;
}

function formatNumeric(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: 4 });
}

function metaRow(label, value) {
  return `<div class="meta-row"><strong>${label}</strong><span>${value}</span></div>`;
}

function metricKey(dataset, metricId, sensorId) { return `${dataset}|${metricId}|${sensorId}`; }

function capitalize(v) { return `${v.charAt(0).toUpperCase()}${v.slice(1)}`; }
