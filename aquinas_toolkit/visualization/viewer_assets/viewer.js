import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const COLOR = {
  deckOldSlab: 0xd6ccb9,
  deckOldWeb: 0xc4baa7,
  deckNewSlab: 0xc9d6d7,
  deckNewWeb: 0xafc2c4,
  deckEdge: 0x28384d,
  pier: 0x7a756f,
  gridMinor: 0xcbd5e1,
  gridMajor: 0x94a3b8,
  accRed: 0xb0323d,
  strBlue: 0x275d9b,
  highAmber: 0xd17a17,
  selected: 0x102c53,
  groundText: 0x7b8da7,
};

const HEALTH_METRIC_ID = "health_anomaly";

const CAM_DEFAULT = { x: 24, y: 12.5, z: -34 };
const CAM_TARGET = { x: 45, y: -1.2, z: 0 };
const GROUND_Y = -18;
const GRID_SIZE = 150;
const GRID_MINOR_STEP = 5;
const GRID_MAJOR_STEP = 15;

const state = {
  manifest: null,
  geometry: null,
  sensorLayout: [],
  metrics: [],
  trends: [],
  eventGroups: [],
  correlations: [],
  reportScores: null,
  metricLookup: new Map(),
  trendLookup: new Map(),
  correlationLookup: new Map(),
  eventLookup: new Map(),
  waveformCache: new Map(),
  selectedDataset: null,
  selectedMetric: "mean_range",
  family: "ALL",
  viewMode: "compact",
  compareMode: "single",
  wireframe: false,
  showCorrelations: false,
  selectedSensorId: null,
  isolatedSensorId: null,
  selectedPreviewEventId: null,
  activeTab: "scene",
  filters: {
    deck: new Set(),
    span: new Set(),
    side: new Set(),
    section: new Set(),
    axis_or_fibre: new Set(),
  },
};

let renderer, scene, camera, controls, raycaster, pointer;
const deckGroups = {};
const pierGroups = {};
const deckMaterials = [];
const deckEdgeMaterials = [];
const sensorObjs = new Map();
const pickTargets = [];
let hoveredId = null;
let minorGrid;
let majorGrid;
let selectionLight = null;

const el = {
  datasetSelect: document.getElementById("dataset-select"),
  metricSelect: document.getElementById("metric-select"),
  familyToggle: document.getElementById("family-toggle"),
  compareMode: document.getElementById("compare-mode"),
  wireframeToggle: document.getElementById("wireframe-toggle"),
  corrToggle: document.getElementById("correlation-toggle"),
  filterGroups: document.getElementById("filter-groups"),
  resetFilters: document.getElementById("reset-filters"),
  resetCamera: document.getElementById("reset-camera"),
  stageTitle: document.getElementById("stage-title"),
  stageSub: document.getElementById("stage-subtitle"),
  analysisSub: document.getElementById("analysis-subtitle"),
  statusStrip: document.getElementById("status-strip"),
  sceneWrap: document.getElementById("scene-wrap"),
  canvas: document.getElementById("bridge-canvas"),
  tooltip: document.getElementById("tooltip"),
  inspector: document.getElementById("inspector"),
  datasetStrip: document.getElementById("dataset-strip"),
  healthCard: document.getElementById("health-card"),
  sceneSelection: document.getElementById("scene-selection"),
  tabButtons: Array.from(document.querySelectorAll("[data-tab]")),
  tabPanels: Array.from(document.querySelectorAll("[data-tab-panel]")),
};

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

  const [geometry, sensorLayout, metrics, trends, eventGroups, correlations, reportScores] =
    await Promise.all([
      fetchJson(files.bridge_geometry),
      fetchJson(files.sensor_layout),
      fetchJson(files.sensor_metrics),
      fetchJson(files.sensor_trends),
      fetchJson(files.event_groups),
      fetchJson(files.correlations),
      fetchJson("./report_scores.json").catch(() => null),
    ]);

  state.geometry = geometry;
  state.sensorLayout = sensorLayout;
  state.metrics = metrics;
  state.trends = trends;
  state.eventGroups = eventGroups;
  state.correlations = correlations;
  state.reportScores = reportScores;
  state.selectedDataset = state.manifest.default_dataset;

  injectHealthScores();
  buildLookups();
  buildControls();
  bindStaticEvents();
  initThree();
  buildBridgeScene();
  render();
  exposeViewerApi();
}

// Small automation hook: lets screenshot/orbit tooling drive the camera and
// switch SET/metric from outside the module closure. Harmless in normal use.
function exposeViewerApi() {
  window.__viewer = {
    get camera() {
      return camera;
    },
    get controls() {
      return controls;
    },
    render,
    resetCamera,
    setDataset(dataset) {
      state.selectedDataset = dataset;
      el.datasetSelect.value = dataset;
      render();
    },
    setMetric(metricId) {
      state.selectedMetric = metricId;
      el.metricSelect.value = metricId;
      render();
    },
    orbit(angleDeg, { radius = 46, height = 12 } = {}) {
      const target = controls.target;
      const a = (angleDeg * Math.PI) / 180;
      camera.position.set(
        target.x + radius * Math.cos(a),
        target.y + height,
        target.z + radius * Math.sin(a)
      );
      controls.update();
    },
  };
}

function meters(value) {
  return value * state.geometry.world.meters_per_normalized_unit;
}

function pointFromRecord(point) {
  return new THREE.Vector3(meters(point.x), meters(point.y), meters(point.z));
}

function vectorFromRecord(vector) {
  return new THREE.Vector3(vector.x, vector.y, vector.z).normalize();
}

function buildLookups() {
  state.metricLookup = new Map(
    state.metrics.map((row) => [metricKey(row.dataset, row.metric_id, row.sensor_id), row])
  );
  state.trendLookup = new Map(state.trends.map((row) => [`${row.sensor_id}|${row.metric_id}`, row]));

  state.correlationLookup = new Map();
  for (const row of state.correlations) {
    const key = `${row.sensor_id}|${row.metric_id}`;
    const current = state.correlationLookup.get(key) || [];
    current.push(row);
    state.correlationLookup.set(key, current);
  }

  state.eventLookup = new Map();
  for (const row of state.eventGroups) {
    const key = `${row.dataset}|${row.deck}`;
    const current = state.eventLookup.get(key) || [];
    current.push(row);
    state.eventLookup.set(key, current);
  }
}

// Report-derived health scoring. The Dataset selector doubles as the SET
// selector; per-SET scores come from the committed report_scores.json so the
// viewer echoes the two-page report without re-running the pipeline.
function injectHealthScores() {
  if (!state.reportScores || !state.reportScores.sets) return;
  const sets = state.reportScores.sets;

  const healthMetric = {
    metric_id: HEALTH_METRIC_ID,
    label: "Health anomaly %",
    description:
      "Percentage of events exceeding the 99th-percentile SET1 reconstruction-error threshold. Modality-level.",
    units_by_family: { ACC: "%", STR: "%" },
    source: "report",
  };
  state.manifest.metric_catalog = [healthMetric, ...state.manifest.metric_catalog];

  for (const dataset of state.manifest.available_datasets) {
    const scores = scoresForDataset(dataset.dataset);
    if (!scores) continue;
    for (const sensor of state.sensorLayout) {
      const value = sensor.measurement_family === "ACC" ? scores.anom_acc : scores.anom_str;
      state.metrics.push({
        dataset: dataset.dataset,
        metric_id: HEALTH_METRIC_ID,
        sensor_id: sensor.sensor_id,
        measurement_family: sensor.measurement_family,
        value,
        unit: "%",
        status_band: "mid",
        source_stage: "report_scores",
      });
    }
  }

  // Open the viewer in health mode when report scores are available.
  state.selectedMetric = HEALTH_METRIC_ID;
}

function setTokenFor(datasetName) {
  const match = /SET\s*0*([0-9]+)/i.exec(datasetName || "");
  return match ? `SET${match[1]}` : null;
}

function scoresForDataset(datasetName) {
  if (!state.reportScores || !state.reportScores.sets) return null;
  const token = setTokenFor(datasetName);
  return token ? state.reportScores.sets[token] ?? null : null;
}

function healthColorFor(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 0x94a3b8;
  const v = Number(value);
  if (v <= 1.5) return 0x2f8f5b; // near reference
  if (v <= 3.0) return 0x7fa53f;
  if (v <= 4.5) return 0xd8a11a;
  if (v <= 6.0) return 0xdd7a17;
  return 0xb0323d; // most deviated
}

function buildControls() {
  el.datasetSelect.innerHTML = state.manifest.available_datasets
    .map((dataset) => `<option value="${dataset.dataset}">${dataset.label} · ${dataset.dataset}</option>`)
    .join("");
  el.datasetSelect.value = state.selectedDataset;

  el.metricSelect.innerHTML = state.manifest.metric_catalog
    .map((metric) => `<option value="${metric.metric_id}">${metric.label}</option>`)
    .join("");
  el.metricSelect.value = state.selectedMetric;

  renderSegmented(el.familyToggle, ["ALL", "ACC", "STR"], state.family, (value) => {
    state.family = value;
    reconcileSelection();
    buildFilterControls();
    render();
  });

  buildFilterControls();
  buildDatasetStrip();
  setActiveTab(state.activeTab);
}

function buildFilterControls() {
  const activeFamily = state.family;
  const axisValues = uniqueValues(
    state.sensorLayout
      .filter((sensor) => activeFamily === "ALL" || sensor.measurement_family === activeFamily)
      .map((sensor) => sensor.axis_or_fibre)
  );

  const groups = [
    ["deck", uniqueValues(state.sensorLayout.map((sensor) => sensor.deck))],
    ["span", uniqueValues(state.sensorLayout.map((sensor) => sensor.span))],
    ["side", uniqueValues(state.sensorLayout.map((sensor) => sensor.side))],
    ["section", uniqueValues(state.sensorLayout.map((sensor) => sensor.section))],
    ["axis_or_fibre", axisValues],
  ];

  el.filterGroups.innerHTML = groups
    .map(([name, values]) => {
      const selected = state.filters[name];
      const options = values
        .map((value) => {
          const checked = selected.size === 0 || selected.has(value) ? "checked" : "";
          return `<label>
            <input type="checkbox" data-filter-group="${name}" value="${value}" ${checked} />
            <span>${value}</span>
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
      (dataset) => `<button
        class="dataset-button ${dataset.dataset === state.selectedDataset ? "active" : ""}"
        data-dataset-button="${dataset.dataset}"
        type="button"
      >
        <strong>${dataset.label}</strong>
        <span>${dataset.dataset}</span>
      </button>`
    )
    .join("");
}

function syncDatasetStrip() {
  el.datasetStrip.querySelectorAll("[data-dataset-button]").forEach((button) => {
    button.classList.toggle("active", button.dataset.datasetButton === state.selectedDataset);
  });
}

function bindStaticEvents() {
  el.datasetSelect.addEventListener("change", (event) => {
    state.selectedDataset = event.target.value;
    render();
  });

  el.metricSelect.addEventListener("change", (event) => {
    state.selectedMetric = event.target.value;
    render();
  });

  el.compareMode.addEventListener("change", (event) => {
    state.compareMode = event.target.value;
    render();
  });

  el.wireframeToggle.addEventListener("change", (event) => {
    state.wireframe = event.target.checked;
    applyWireframeMode();
    render();
  });

  el.corrToggle.addEventListener("change", (event) => {
    state.showCorrelations = event.target.checked;
    render();
  });

  el.resetFilters.addEventListener("click", () => {
    for (const key of Object.keys(state.filters)) {
      state.filters[key] = new Set();
    }
    state.isolatedSensorId = null;
    buildFilterControls();
    render();
  });

  el.resetCamera.addEventListener("click", resetCamera);

  el.filterGroups.addEventListener("change", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLInputElement) || !target.dataset.filterGroup) return;
    const group = target.dataset.filterGroup;
    const checked = new Set(
      Array.from(el.filterGroups.querySelectorAll(`input[data-filter-group="${group}"]:checked`))
        .map((input) => input.value)
    );
    const all = new Set(
      Array.from(el.filterGroups.querySelectorAll(`input[data-filter-group="${group}"]`))
        .map((input) => input.value)
    );
    state.filters[group] = checked.size === all.size ? new Set() : checked;
    reconcileSelection();
    render();
  });

  el.datasetStrip.addEventListener("click", (event) => {
    const target = event.target.closest("[data-dataset-button]");
    if (!target) return;
    state.selectedDataset = target.dataset.datasetButton;
    el.datasetSelect.value = state.selectedDataset;
    render();
  });

  el.sceneSelection.addEventListener("click", (event) => {
    const target = event.target.closest("[data-open-analysis]");
    if (target) setActiveTab("analysis");
  });

  for (const button of el.tabButtons) {
    button.addEventListener("click", () => setActiveTab(button.dataset.tab));
  }
}

function setActiveTab(tabId) {
  state.activeTab = tabId;
  for (const button of el.tabButtons) {
    button.classList.toggle("active", button.dataset.tab === tabId);
  }
  for (const panel of el.tabPanels) {
    panel.classList.toggle("active", panel.dataset.tabPanel === tabId);
  }
  if (tabId === "scene") {
    requestAnimationFrame(() => requestAnimationFrame(onResize));
  }
}

function render() {
  const visibleSensors = getVisibleSensors();
  syncDatasetStrip();
  updateHeader(visibleSensors);
  updateHealthCard();
  updateSceneSelection();
  updateSensorGlyphs(visibleSensors);
  renderInspector();
}

function updateHealthCard() {
  if (!el.healthCard) return;
  const scores = scoresForDataset(state.selectedDataset);
  if (!scores) {
    el.healthCard.hidden = true;
    el.healthCard.innerHTML = "";
    return;
  }
  el.healthCard.hidden = false;
  const healthMode = state.selectedMetric === HEALTH_METRIC_ID;
  const scoreColor = healthScoreColor(scores.health_score);
  el.healthCard.style.setProperty("--health-accent", scoreColor);

  el.healthCard.innerHTML = `
    <div class="health-head">
      <span class="health-kicker">${scores.label} · health score</span>
      <div class="health-score" style="color:${scoreColor}">
        ${formatScore(scores.health_score)}<span class="health-score-max">/ 100</span>
      </div>
      <p class="health-note">${scores.note || ""}</p>
    </div>
    <div class="health-rates">
      ${healthRateChip("All", scores.anom_all)}
      ${healthRateChip("Acc", scores.anom_acc)}
      ${healthRateChip("Strain", scores.anom_str)}
    </div>
    ${healthMode ? healthLegendMarkup() : ""}
    <p class="health-provenance">
      Events over the 99th-pct SET1 threshold. SET1 = healthiest reference (100).
      From the EWSHM 2026 report.
    </p>`;
}

function healthScoreColor(score) {
  const cssByHex = healthColorFor(scoreToAnomalyProxy(score));
  return `#${cssByHex.toString(16).padStart(6, "0")}`;
}

// Higher health score = healthier = lower anomaly. Map to the same ramp so the
// headline number and the glyph colors read consistently.
function scoreToAnomalyProxy(score) {
  const s = Number(score);
  if (Number.isNaN(s)) return null;
  return Math.max(0, (100 - s) / 12);
}

function formatScore(score) {
  const s = Number(score);
  if (Number.isNaN(s)) return "n/a";
  return Number.isInteger(s) ? `${s}` : s.toFixed(1);
}

function healthRateChip(label, value) {
  const hex = healthColorFor(value);
  const css = `#${hex.toString(16).padStart(6, "0")}`;
  const text = value === null || value === undefined ? "n/a" : `${Number(value).toFixed(2)}%`;
  return `<div class="health-rate">
    <span class="health-rate-dot" style="background:${css}"></span>
    <span class="health-rate-label">${label}</span>
    <span class="health-rate-value">${text}</span>
  </div>`;
}

function healthLegendMarkup() {
  const stops = [
    ["≤1.5%", 0x2f8f5b],
    ["≤3%", 0x7fa53f],
    ["≤4.5%", 0xd8a11a],
    ["≤6%", 0xdd7a17],
    [">6%", 0xb0323d],
  ];
  const chips = stops
    .map(([label, hex]) => {
      const css = `#${hex.toString(16).padStart(6, "0")}`;
      return `<span class="health-ramp-chip"><i style="background:${css}"></i>${label}</span>`;
    })
    .join("");
  return `<div class="health-ramp"><span class="health-ramp-title">Glyph anomaly %</span>${chips}</div>`;
}

function updateHeader(visibleSensors) {
  const dataset = state.manifest.available_datasets.find(
    (item) => item.dataset === state.selectedDataset
  );
  const datasetLabel = dataset ? dataset.label : state.selectedDataset;
  const metricLabel = labelForMetric(state.selectedMetric);

  el.stageTitle.textContent = datasetLabel;
  el.stageSub.textContent = `${metricLabel} · analytical placement view`;
  el.analysisSub.textContent = state.selectedSensorId
    ? `Detailed view for ${state.selectedSensorId}.`
    : "Select a marker in the 3D view to inspect the sensor in detail.";

  el.statusStrip.innerHTML = [
    `${visibleSensors.length} shown`,
    state.family,
    state.selectedSensorId ? "Selection active" : "No selection",
  ]
    .map((text) => `<span class="status-pill">${text}</span>`)
    .join("");
}

function updateSceneSelection() {
  const sensor = state.sensorLayout.find((item) => item.sensor_id === state.selectedSensorId);
  if (!sensor) {
    el.sceneSelection.innerHTML = `
      <div class="card selection-card compact-empty">
        <span class="selection-kicker">Scene focus</span>
        <div class="selection-compact-text">Click a marker to inspect one sensor.</div>
      </div>`;
    return;
  }

  const metric = getMetric(sensor.sensor_id);
  const local = sensor.local_position;

  el.sceneSelection.innerHTML = `
    <div class="card selection-card">
      <div class="selection-head">
        <span class="selection-kicker">${sensor.deck} · ${sensor.span}_${sensor.section}</span>
        <h3>${sensor.sensor_id}</h3>
        <div class="metric-value">${formatMetric(metric)}</div>
        <p class="card-note">${labelForMetric(state.selectedMetric)} · ${metric?.unit ?? "n/a"}</p>
      </div>
      <div class="selection-meta">
        <div class="selection-meta-grid">
        ${metaRow("Deck", sensor.deck)}
        ${metaRow("Span", sensor.span)}
        ${metaRow("Side", sensor.side)}
        ${metaRow("Surface", humanizeSurface(sensor.mount_surface))}
        ${metaRow("Local X", `${formatNumeric(meters(local.x))} m`)}
        ${metaRow("Local Y", `${formatNumeric(meters(local.y))} m`)}
        </div>
        <button class="ghost-button selection-action" type="button" data-open-analysis>Open analysis</button>
      </div>
    </div>`;
}

function initThree() {
  renderer = new THREE.WebGLRenderer({ canvas: el.canvas, antialias: true, alpha: false });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setClearColor(0xf4f7fb, 1);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0xf4f7fb, 80, 220);

  camera = new THREE.PerspectiveCamera(38, 1, 0.1, 350);
  camera.position.set(CAM_DEFAULT.x, CAM_DEFAULT.y, CAM_DEFAULT.z);

  controls = new OrbitControls(camera, el.canvas);
  controls.target.set(CAM_TARGET.x, CAM_TARGET.y, CAM_TARGET.z);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.minPolarAngle = 0.15;
  controls.maxPolarAngle = Math.PI * 0.48;
  controls.minDistance = 18;
  controls.maxDistance = 150;
  controls.update();

  const ambient = new THREE.AmbientLight(0xffffff, 0.78);
  scene.add(ambient);

  const sun = new THREE.DirectionalLight(0xfff7eb, 1.1);
  sun.position.set(42, 72, 34);
  sun.castShadow = true;
  sun.shadow.mapSize.set(4096, 4096);
  sun.shadow.camera.near = 1;
  sun.shadow.camera.far = 180;
  sun.shadow.camera.left = -45;
  sun.shadow.camera.right = 55;
  sun.shadow.camera.top = 55;
  sun.shadow.camera.bottom = -55;
  sun.shadow.bias = -0.0004;
  scene.add(sun);

  const fill = new THREE.DirectionalLight(0xdbeafe, 0.45);
  fill.position.set(-36, 26, -52);
  scene.add(fill);

  const backFill = new THREE.DirectionalLight(0xfff0d0, 0.22);
  backFill.position.set(-20, -6, 55);
  scene.add(backFill);

  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(220, 160),
    new THREE.ShadowMaterial({ opacity: 0.06 })
  );
  ground.rotation.x = -Math.PI / 2;
  ground.position.set(45, GROUND_Y, 0);
  ground.receiveShadow = true;
  scene.add(ground);

  raycaster = new THREE.Raycaster();
  pointer = new THREE.Vector2();

  el.canvas.addEventListener("pointermove", onPointerMove);
  el.canvas.addEventListener("click", onCanvasClick);
  el.canvas.addEventListener("dblclick", onCanvasDoubleClick);

  const observer = new ResizeObserver(onResize);
  observer.observe(el.sceneWrap);
  onResize();

  animate();
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

function onResize() {
  const width = el.sceneWrap.clientWidth || 800;
  const height = el.sceneWrap.clientHeight || 480;
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}

function resetCamera() {
  camera.position.set(CAM_DEFAULT.x, CAM_DEFAULT.y, CAM_DEFAULT.z);
  controls.target.set(CAM_TARGET.x, CAM_TARGET.y, CAM_TARGET.z);
  controls.update();
}

function buildBridgeScene() {
  addGroundContext();
  buildDecks();
  buildPiers();
  addDirectionLabels();
}

function addGroundContext() {
  const minorDivisions = GRID_SIZE / GRID_MINOR_STEP;
  const majorDivisions = GRID_SIZE / GRID_MAJOR_STEP;

  minorGrid = new THREE.GridHelper(GRID_SIZE, minorDivisions, COLOR.gridMinor, COLOR.gridMinor);
  minorGrid.position.set(45, GROUND_Y + 0.01, 0);
  setHelperOpacity(minorGrid, 0.14);
  scene.add(minorGrid);

  majorGrid = new THREE.GridHelper(GRID_SIZE, majorDivisions, COLOR.gridMajor, COLOR.gridMajor);
  majorGrid.position.set(45, GROUND_Y + 0.02, 0);
  setHelperOpacity(majorGrid, 0.22);
  scene.add(majorGrid);

  const tickGroup = new THREE.Group();
  const tickMaterial = new THREE.LineBasicMaterial({ color: COLOR.gridMajor, transparent: true, opacity: 0.32 });
  for (let value = 0; value <= 90; value += GRID_MAJOR_STEP) {
    const geometry = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(value, GROUND_Y + 0.02, 18),
      new THREE.Vector3(value, GROUND_Y + 0.02, 20),
    ]);
    tickGroup.add(new THREE.Line(geometry, tickMaterial));
    tickGroup.add(createGroundText(`${value} m`, {
      width: 5.6,
      height: 1.3,
      fontSize: 44,
      color: "#74849a",
      opacity: 0.7,
      position: new THREE.Vector3(value, GROUND_Y + 0.04, 22),
    }));
  }

  tickGroup.add(createGroundText("Span 1", {
    width: 7.5,
    height: 1.6,
    fontSize: 52,
    color: "#8da0b8",
    opacity: 0.42,
    position: new THREE.Vector3(22.5, GROUND_Y + 0.04, 15.5),
  }));
  tickGroup.add(createGroundText("Span 2", {
    width: 7.5,
    height: 1.6,
    fontSize: 52,
    color: "#8da0b8",
    opacity: 0.42,
    position: new THREE.Vector3(67.5, GROUND_Y + 0.04, 15.5),
  }));
  scene.add(tickGroup);
}

function buildDecks() {
  const viewModes = state.geometry.view_modes;
  const cross = state.geometry.cross_section;

  const depth = meters(cross.depth);
  const topWidth = meters(cross.top_slab_width);
  const bottomWidth = meters(cross.bottom_slab_width);
  const slabThickness = meters(cross.slab_thickness);
  const webOuterTopHalf = meters(cross.web_outer_top_width / 2);
  const webOuterBottomHalf = meters(cross.web_outer_bottom_width / 2);
  const webInnerTopHalf = meters(cross.web_inner_top_width / 2);
  const webInnerBottomHalf = meters(cross.web_inner_bottom_width / 2);
  const halfDepth = depth / 2;
  const topSlabCenterY = halfDepth - (slabThickness / 2);
  const bottomSlabCenterY = -halfDepth + (slabThickness / 2);
  const webTopY = halfDepth - slabThickness;
  const webBottomY = -halfDepth + slabThickness;

  for (const deckData of state.geometry.deck_meshes) {
    const group = new THREE.Group();
    deckGroups[deckData.deck] = group;
    scene.add(group);

    const deckCenterZ = meters(viewModes[state.viewMode].deck_centers[deckData.deck]);
    group.position.z = deckCenterZ;

    const slabColor = deckData.deck === "OLD" ? COLOR.deckOldSlab : COLOR.deckNewSlab;
    const webColor = deckData.deck === "OLD" ? COLOR.deckOldWeb : COLOR.deckNewWeb;

    for (const segment of deckData.segments) {
      const startX = meters(segment.x_start);
      const length = meters(segment.x_end - segment.x_start);
      const centerX = startX + (length / 2);

      const slabMaterial = new THREE.MeshStandardMaterial({
        color: slabColor,
        roughness: 0.86,
        metalness: 0.02,
        transparent: true,
        opacity: 0.78,
      });
      deckMaterials.push({ material: slabMaterial, opacity: 0.78 });
      const webMaterial = new THREE.MeshStandardMaterial({
        color: webColor,
        roughness: 0.9,
        metalness: 0.02,
        transparent: true,
        opacity: 0.74,
        side: THREE.DoubleSide,
      });
      deckMaterials.push({ material: webMaterial, opacity: 0.74 });

      const topSlab = new THREE.Mesh(
        new THREE.BoxGeometry(length, slabThickness, topWidth),
        slabMaterial
      );
      topSlab.position.set(centerX, topSlabCenterY, 0);
      topSlab.castShadow = true;
      topSlab.receiveShadow = true;
      group.add(topSlab);
      addEdgeOverlay(topSlab.geometry, topSlab.position, topSlab.rotation, group);

      const bottomSlab = new THREE.Mesh(
        new THREE.BoxGeometry(length, slabThickness, bottomWidth),
        webMaterial
      );
      bottomSlab.position.set(centerX, bottomSlabCenterY, 0);
      bottomSlab.castShadow = true;
      bottomSlab.receiveShadow = true;
      group.add(bottomSlab);
      addEdgeOverlay(bottomSlab.geometry, bottomSlab.position, bottomSlab.rotation, group);

      const webProfiles = [
        [
          [-webOuterTopHalf, webTopY],
          [-webInnerTopHalf, webTopY],
          [-webInnerBottomHalf, webBottomY],
          [-webOuterBottomHalf, webBottomY],
        ],
        [
          [webInnerTopHalf, webTopY],
          [webOuterTopHalf, webTopY],
          [webOuterBottomHalf, webBottomY],
          [webInnerBottomHalf, webBottomY],
        ],
      ];

      for (const profile of webProfiles) {
        const shape = new THREE.Shape();
        shape.moveTo(profile[0][0], profile[0][1]);
        for (const [z, y] of profile.slice(1)) {
          shape.lineTo(z, y);
        }
        shape.closePath();

        const geometry = new THREE.ExtrudeGeometry(shape, { depth: length, bevelEnabled: false });
        const web = new THREE.Mesh(geometry, webMaterial);
        web.rotation.y = Math.PI / 2;
        web.position.set(startX, 0, 0);
        web.castShadow = true;
        web.receiveShadow = true;
        group.add(web);
        addEdgeOverlay(geometry, web.position, web.rotation, group);
      }
    }

    addDeckLabel(group, deckData.deck);
  }
}

function addDeckLabel(group, deckName) {
  const halfDepth = meters(state.geometry.cross_section.depth) / 2;
  const color = deckName === "OLD" ? "#6b5b45" : "#2f5f5d";

  // Text painted flat on the top slab, reading along the bridge. A single
  // upward-facing (FrontSide) plane avoids the mirrored back-face artifact,
  // and a non-flipped texture keeps the letters the right way round.
  const canvas = createTextCanvas(`${deckName} deck`, {
    width: 1024,
    height: 160,
    fontSize: 104,
    color,
    paddingX: 60,
  });
  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;

  const plane = new THREE.Mesh(
    new THREE.PlaneGeometry(30, 4.7),
    new THREE.MeshBasicMaterial({
      map: texture,
      transparent: true,
      opacity: 0.92,
      depthWrite: false,
      side: THREE.DoubleSide,
    })
  );
  // Lay the plane flat on top of the slab, letters reading along +x.
  plane.rotation.x = -Math.PI / 2;
  plane.position.set(24, halfDepth + 0.08, 0);
  plane.renderOrder = 4;
  group.add(plane);
}

function pierX(x) {
  // Tuck the end supports just inside the deck ends so the column sits under
  // the deck rather than half-past its edge; leave interior piers where they are.
  const bridgeLength = meters(state.geometry.world.bridge_length_m / state.geometry.world.meters_per_normalized_unit);
  if (x <= 1) return x + 2.5;
  if (x >= bridgeLength - 1) return x - 2.5;
  return x;
}

function buildPiers() {
  const cross = state.geometry.cross_section;
  const halfDepth = meters(cross.depth) / 2;
  const deckWidth = meters(cross.top_slab_width);
  const padHeight = 0.7;
  const footingHeight = 1.0;
  const groundY = GROUND_Y;
  const viewModes = state.geometry.view_modes;

  for (const deckData of state.geometry.deck_meshes) {
    const group = new THREE.Group();
    group.position.z = meters(viewModes[state.viewMode].deck_centers[deckData.deck]);
    pierGroups[deckData.deck] = group;
    scene.add(group);

    // Every support is the same round column: pad + tall column + footing,
    // sitting on the ground and centered under the deck.
    const columnHeight = -halfDepth - groundY - padHeight - footingHeight;
    for (const pier of state.geometry.pier_anchors) {
      const x = pierX(meters(pier.x));
      const material = new THREE.MeshStandardMaterial({ color: COLOR.pier, roughness: 0.92 });

      const pad = new THREE.Mesh(new THREE.BoxGeometry(4.8, padHeight, deckWidth * 0.5), material);
      pad.position.set(x, -(halfDepth + padHeight / 2), 0);
      pad.castShadow = true;
      group.add(pad);

      const column = new THREE.Mesh(
        new THREE.CylinderGeometry(1.05, 1.5, columnHeight, 16),
        material
      );
      column.position.set(x, -(halfDepth + padHeight + columnHeight / 2), 0);
      column.castShadow = true;
      group.add(column);

      const footing = new THREE.Mesh(new THREE.BoxGeometry(5.4, footingHeight, 5.4), material);
      footing.position.set(x, groundY + footingHeight / 2, 0);
      footing.castShadow = true;
      group.add(footing);
    }
  }
}

function addDirectionLabels() {
  const labels = [
    { text: "UPSTREAM", color: "#7b8da7", z: 20 },
    { text: "DOWNSTREAM", color: "#8aa8bb", z: -20 },
  ];

  for (const label of labels) {
    scene.add(createGroundText(label.text, {
      width: 18,
      height: 3.2,
      fontSize: 68,
      color: label.color,
      opacity: 0.74,
      position: new THREE.Vector3(70, GROUND_Y + 0.05, label.z),
    }));
  }
}

function addEdgeOverlay(geometry, position, rotation, parent) {
  const material = new THREE.LineBasicMaterial({
    color: COLOR.deckEdge,
    transparent: true,
    opacity: 0.32,
  });
  deckEdgeMaterials.push({ material, opacity: 0.32 });
  const edges = new THREE.LineSegments(
    new THREE.EdgesGeometry(geometry, 18),
    material
  );
  edges.position.copy(position);
  edges.rotation.copy(rotation);
  parent.add(edges);
}

function createTextCanvas(
  text,
  {
    width = 512,
    height = 128,
    fontSize = 48,
    color = "#102c53",
    opacity = 1,
    backgroundColor = null,
    borderColor = null,
    radius = 0,
    paddingX = 0,
  }
) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;

  const context = canvas.getContext("2d");
  context.clearRect(0, 0, width, height);

  if (backgroundColor) {
    context.globalAlpha = opacity;
    context.fillStyle = backgroundColor;
    drawRoundedRect(context, 0, 0, width, height, radius);
    context.fill();

    if (borderColor) {
      context.strokeStyle = borderColor;
      context.lineWidth = 4;
      context.stroke();
    }
  }

  context.globalAlpha = opacity;
  context.font = `800 ${fontSize}px Manrope`;
  context.fillStyle = color;
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(text, width / 2, height / 2, width - (paddingX * 2));
  return canvas;
}

function drawRoundedRect(context, x, y, width, height, radius) {
  const limited = Math.min(radius, width / 2, height / 2);
  context.beginPath();
  context.moveTo(x + limited, y);
  context.lineTo(x + width - limited, y);
  context.quadraticCurveTo(x + width, y, x + width, y + limited);
  context.lineTo(x + width, y + height - limited);
  context.quadraticCurveTo(x + width, y + height, x + width - limited, y + height);
  context.lineTo(x + limited, y + height);
  context.quadraticCurveTo(x, y + height, x, y + height - limited);
  context.lineTo(x, y + limited);
  context.quadraticCurveTo(x, y, x + limited, y);
  context.closePath();
}

function setHelperOpacity(helper, opacity) {
  const materials = Array.isArray(helper.material) ? helper.material : [helper.material];
  for (const material of materials) {
    material.transparent = true;
    material.opacity = opacity;
  }
}

function createGroundText(text, { width, height, fontSize, color, opacity, position }) {
  const texture = new THREE.CanvasTexture(createTextCanvas(text, { fontSize, color, opacity }));
  texture.needsUpdate = true;
  const material = new THREE.MeshBasicMaterial({
    map: texture,
    transparent: true,
    depthWrite: false,
  });
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(width, height), material);
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.copy(position);
  return mesh;
}


function updateSensorGlyphs(visibleSensors) {
  const visible = new Set(visibleSensors.map((sensor) => sensor.sensor_id));

  for (const [sensorId, object] of sensorObjs) {
    if (!visible.has(sensorId)) {
      if (selectionLight && selectionLight.parent === object.group) {
        object.group.remove(selectionLight);
        selectionLight = null;
      }
      scene.remove(object.group);
      sensorObjs.delete(sensorId);
      const index = pickTargets.indexOf(object.pickMesh);
      if (index !== -1) pickTargets.splice(index, 1);
    }
  }

  for (const sensor of visibleSensors) {
    const metric = getMetric(sensor.sensor_id);
    const isSelected = sensor.sensor_id === state.selectedSensorId;
    const isHigh = metric?.status_band === "high";
    const isLow = metric?.status_band === "low";

    if (sensorObjs.has(sensor.sensor_id)) {
      const object = sensorObjs.get(sensor.sensor_id);
      const color = glyphColor(sensor, isHigh, isSelected, metric);
      for (const mesh of object.meshes) {
        mesh.material.color.setHex(color);
        mesh.material.emissive?.setHex(isSelected ? 0x143763 : 0x000000);
        mesh.material.opacity = isLow ? 0.34 : 1.0;
      }
      const baseScale = state.wireframe ? 1.4 : 1.0;
      object.group.scale.setScalar(
        isSelected ? baseScale * 1.18 : sensor.sensor_id === hoveredId ? baseScale * 1.08 : baseScale
      );
      if (object.group.userData.labelSprite) {
        object.group.userData.labelSprite.visible = state.wireframe;
      }
    } else {
      const group = buildGlyph(sensor, metric, isSelected);
      const local = sensor.local_position;
      const deckCenter = deckGroups[sensor.deck]?.position.z
        ?? meters(state.geometry.view_modes[state.viewMode].deck_centers[sensor.deck]);
      group.position.set(meters(local.x), meters(local.y), deckCenter + meters(local.z));
      scene.add(group);

      const pickBounds = glyphPickBounds(group);
      const pickMesh = new THREE.Mesh(
        new THREE.SphereGeometry(pickBounds.radius, 10, 10),
        new THREE.MeshBasicMaterial({ visible: false })
      );
      pickMesh.userData.sensorId = sensor.sensor_id;
      pickMesh.position.copy(pickBounds.center);
      group.add(pickMesh);
      pickTargets.push(pickMesh);

      sensorObjs.set(sensor.sensor_id, {
        group,
        meshes: group.userData.coloredMeshes,
        pickMesh,
      });
    }
  }

  syncSelectionLight();
}

function glyphColor(sensor, isHigh, isSelected, metric) {
  if (isSelected) return COLOR.selected;
  if (state.selectedMetric === HEALTH_METRIC_ID) return healthColorFor(metric?.value);
  if (isHigh) return COLOR.highAmber;
  return sensor.measurement_family === "ACC" ? COLOR.accRed : COLOR.strBlue;
}

function buildGlyph(sensor, metric, isSelected) {
  const group = new THREE.Group();
  const meshes = [];
  const isHigh = metric?.status_band === "high";

  const material = () => new THREE.MeshPhysicalMaterial({
    color: glyphColor(sensor, isHigh, isSelected, metric),
    roughness: 0.28,
    metalness: 0.08,
    clearcoat: 0.4,
    clearcoatRoughness: 0.18,
    emissive: isSelected ? new THREE.Color(0x143763) : new THREE.Color(0x000000),
    transparent: true,
    opacity: metric?.status_band === "low" ? 0.34 : 1.0,
  });

  const size = 0.78;

  if (sensor.measurement_family === "ACC") {
    const shaft = new THREE.Mesh(
      new THREE.CylinderGeometry(size * 0.16, size * 0.16, size * 1.35, 10),
      material()
    );
    shaft.position.y = size * 0.55;
    const head = new THREE.Mesh(
      new THREE.ConeGeometry(size * 0.34, size * 0.55, 10),
      material()
    );
    head.position.y = size * 1.42;
    meshes.push(shaft, head);
    group.add(shaft, head);

    const targetAxis = vectorFromRecord(sensor.glyph_orientation);
    group.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), targetAxis);
  } else {
    const body = new THREE.Mesh(
      new THREE.CylinderGeometry(size * 0.12, size * 0.12, size * 1.2, 10),
      material()
    );
    body.rotation.z = Math.PI / 2;
    const headLeft = new THREE.Mesh(
      new THREE.ConeGeometry(size * 0.28, size * 0.42, 10),
      material()
    );
    headLeft.rotation.z = Math.PI / 2;
    headLeft.position.x = -size * 0.88;
    const headRight = new THREE.Mesh(
      new THREE.ConeGeometry(size * 0.28, size * 0.42, 10),
      material()
    );
    headRight.rotation.z = -Math.PI / 2;
    headRight.position.x = size * 0.88;
    meshes.push(body, headLeft, headRight);
    group.add(body, headLeft, headRight);

    const targetAxis = vectorFromRecord(sensor.glyph_orientation);
    group.quaternion.setFromUnitVectors(new THREE.Vector3(1, 0, 0), targetAxis);
  }

  const labelTexture = new THREE.CanvasTexture(
    createTextCanvas(sensor.section, {
      width: 360,
      height: 96,
      fontSize: 40,
      color: "#102c53",
      backgroundColor: "#ffffff",
      borderColor: "#d8e0ea",
      radius: 44,
      paddingX: 24,
    })
  );
  labelTexture.needsUpdate = true;
  const labelSprite = new THREE.Sprite(
    new THREE.SpriteMaterial({
      map: labelTexture,
      transparent: true,
      depthTest: false,
      depthWrite: false,
    })
  );
  labelSprite.scale.set(4.5, 1.2, 1);
  labelSprite.position.set(0, 1.45, 0);
  labelSprite.visible = false;
  group.add(labelSprite);

  group.userData.coloredMeshes = meshes;
  group.userData.labelSprite = labelSprite;
  for (const mesh of meshes) {
    mesh.castShadow = true;
  }
  const baseScale = state.wireframe ? 1.4 : 1.0;
  group.scale.setScalar(isSelected ? baseScale * 1.18 : baseScale);
  return group;
}

function glyphPickBounds(group) {
  const box = new THREE.Box3().setFromObject(group);
  const sphere = box.getBoundingSphere(new THREE.Sphere());
  return {
    center: sphere.center,
    radius: Math.max(sphere.radius, 0.55),
  };
}

function getCanvasPointer(event) {
  const rect = el.canvas.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / rect.width) * 2 - 1,
    y: -((event.clientY - rect.top) / rect.height) * 2 + 1,
  };
}

function raycastSensor(event) {
  const point = getCanvasPointer(event);
  pointer.set(point.x, point.y);
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObjects(pickTargets, false);
  return hits.length ? hits[0].object.userData.sensorId : null;
}

function onPointerMove(event) {
  const sensorId = raycastSensor(event);
  const baseScale = state.wireframe ? 1.4 : 1.0;
  if (sensorId !== hoveredId) {
    if (hoveredId && sensorObjs.has(hoveredId)) {
      const object = sensorObjs.get(hoveredId);
      object.group.scale.setScalar(hoveredId === state.selectedSensorId ? baseScale * 1.18 : baseScale);
    }
    hoveredId = sensorId;
    if (hoveredId && sensorObjs.has(hoveredId) && hoveredId !== state.selectedSensorId) {
      sensorObjs.get(hoveredId).group.scale.setScalar(baseScale * 1.08);
    }
  }

  if (!sensorId) {
    el.tooltip.classList.add("hidden");
    return;
  }

  const sensor = state.sensorLayout.find((item) => item.sensor_id === sensorId);
  const metric = getMetric(sensorId);
  el.tooltip.innerHTML = `
    <strong>${sensorId}</strong><br>
    ${sensor.deck} · ${sensor.span}_${sensor.section} · ${sensor.side}<br>
    ${humanizeSurface(sensor.mount_surface)}<br>
    ${labelForMetric(state.selectedMetric)}: ${formatMetric(metric)}`;
  el.tooltip.classList.remove("hidden");
  el.tooltip.style.left = `${event.offsetX}px`;
  el.tooltip.style.top = `${event.offsetY}px`;
}

function onCanvasClick(event) {
  const sensorId = raycastSensor(event);
  if (!sensorId) {
    state.selectedSensorId = null;
    state.selectedPreviewEventId = null;
    syncSelectionLight();
    render();
    return;
  }
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

function renderInspector() {
  const sensor = state.sensorLayout.find((item) => item.sensor_id === state.selectedSensorId);
  if (!sensor) {
    el.inspector.innerHTML = `
      <div class="card">
        <h3>No sensor selected</h3>
        <p class="card-note">
          Use the 3D view to select a sensor. The analysis tab will show its metric value,
          location, homologous comparison, trend, correlations, and waveform preview.
        </p>
      </div>`;
    return;
  }

  const metric = getMetric(sensor.sensor_id);
  const homologous = sensor.homologous_sensor_id
    ? state.sensorLayout.find((item) => item.sensor_id === sensor.homologous_sensor_id)
    : null;
  const homologousMetric = homologous ? getMetric(homologous.sensor_id) : null;
  const relatedSensors = getRelatedSensors(sensor);
  const correlations = (
    state.correlationLookup.get(`${sensor.sensor_id}|${state.selectedMetric}`) || []
  ).slice(0, 4);
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
        ${metaRow("Surface", humanizeSurface(sensor.mount_surface))}
        ${metaRow("Axis / fibre", sensor.axis_or_fibre)}
      </div>
    </div>

    <div class="card">
      <h3>Mount geometry</h3>
      <div class="meta-grid">
        ${metaRow("Anchor X", `${formatNumeric(meters(sensor.anchor_local.x))} m`)}
        ${metaRow("Anchor Y", `${formatNumeric(meters(sensor.anchor_local.y))} m`)}
        ${metaRow("Anchor Z", `${formatNumeric(meters(sensor.anchor_local.z))} m`)}
        ${metaRow("Offset", `${formatNumeric(meters(sensor.readability_offset))} m`)}
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
               .map((point) => `${point.dataset_label}: ${formatNumeric(point.value)}`)
               .join(" · ")}</p>`
          : `<p class="empty-state">No trend points available for this metric.</p>`
      }
    </div>

    <div class="card">
      <h3>Family-aware comparisons</h3>
      ${
        relatedSensors.length
          ? `<ul class="mini-list">${relatedSensors
              .map((entry) => `<li>${entry.sensor_id} · ${formatMetric(getMetric(entry.sensor_id))}</li>`)
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
                (row) => `<li>${row.target_sensor_id} · r=${formatNumeric(row.correlation)}${
                  row.cross_deck_homologous ? " · homologous" : ""
                }</li>`
              )
              .join("")}</ul>`
          : `<p class="empty-state">Correlations are derived from proxy metric trends.</p>`
      }
    </div>

    ${waveformCard}
  `;

  const previewSelect = el.inspector.querySelector("[data-waveform-select]");
  if (previewSelect) {
    previewSelect.addEventListener("change", async (event) => {
      const selectedId = event.target.value;
      if (!selectedId) {
        state.selectedPreviewEventId = null;
        renderInspector();
        return;
      }
      state.selectedPreviewEventId = selectedId;
      await ensureWaveformLoaded(selectedId);
      renderInspector();
    });
  }
}

function renderWaveformCard(sensor) {
  const events = (state.eventLookup.get(`${state.selectedDataset}|${sensor.deck}`) || [])
    .filter((event) => event.sensor_ids.includes(sensor.sensor_id));

  if (!events.length) {
    return `<div class="card">
      <h3>Deck-scoped event detail</h3>
      <p class="empty-state">No event groups exported for this deck and dataset.</p>
    </div>`;
  }

  const selectable = events.filter((event) => event.waveform_preview_path);
  const activeId =
    selectable.find((event) => event.event_group_id === state.selectedPreviewEventId)?.event_group_id
    || selectable[0]?.event_group_id
    || "";
  const waveformData = activeId ? state.waveformCache.get(activeId) : null;
  const trace = waveformData?.traces?.find((item) => item.sensor_id === sensor.sensor_id);

  return `<div class="card">
    <h3>Deck-scoped event detail</h3>
    <p class="card-note">Events are keyed by dataset + deck + time window.</p>
    ${
      selectable.length
        ? `<div class="control">
            <label for="waveform-select">Preview event</label>
            <select id="waveform-select" data-waveform-select>
              ${selectable
                .map(
                  (event) => `<option value="${event.event_group_id}" ${event.event_group_id === activeId ? "selected" : ""}>
                    ${event.start_time_utc} · ${event.sensor_count} sensors
                  </option>`
                )
                .join("")}
            </select>
          </div>`
        : ""
    }
    ${
      !selectable.length
        ? `<p class="empty-state">Waveform previews were not included in this bundle.</p>`
        : trace
        ? renderWaveform(trace)
        : `<p class="empty-state">Select an event to inspect this sensor's waveform.</p>`
    }
  </div>`;
}

async function ensureWaveformLoaded(eventGroupId) {
  if (state.waveformCache.has(eventGroupId)) return;
  const event = state.eventGroups.find((item) => item.event_group_id === eventGroupId);
  if (!event?.waveform_preview_path) return;
  const payload = await fetchJson(event.waveform_preview_path);
  state.waveformCache.set(eventGroupId, payload);
}

async function ensureDefaultPreview() {
  const sensor = state.sensorLayout.find((item) => item.sensor_id === state.selectedSensorId);
  if (!sensor) return;
  const event = (state.eventLookup.get(`${state.selectedDataset}|${sensor.deck}`) || []).find(
    (entry) => entry.waveform_preview_path && entry.sensor_ids.includes(sensor.sensor_id)
  );
  if (!event) return;
  state.selectedPreviewEventId = event.event_group_id;
  await ensureWaveformLoaded(event.event_group_id);
}

function renderSparkline(points) {
  const valid = points.filter((point) => point.value !== null && point.value !== undefined);
  if (!valid.length) return "";

  const width = 250;
  const height = 100;
  const min = Math.min(...valid.map((point) => point.value));
  const max = Math.max(...valid.map((point) => point.value));
  const range = max - min || 1;

  const path = valid.map((point, index) => {
    const x = 14 + (index * (width - 28)) / Math.max(valid.length - 1, 1);
    const y = height - 14 - ((point.value - min) / range) * (height - 28);
    return `${index === 0 ? "M" : "L"} ${x} ${y}`;
  }).join(" ");

  const area = `M 14 ${height - 14} `
    + valid.map((point, index) => {
      const x = 14 + (index * (width - 28)) / Math.max(valid.length - 1, 1);
      const y = height - 14 - ((point.value - min) / range) * (height - 28);
      return `L ${x} ${y}`;
    }).join(" ")
    + ` L ${14 + ((valid.length - 1) * (width - 28)) / Math.max(valid.length - 1, 1)} ${height - 14} Z`;

  const dots = valid.map((point, index) => {
    const x = 14 + (index * (width - 28)) / Math.max(valid.length - 1, 1);
    const y = height - 14 - ((point.value - min) / range) * (height - 28);
    return `<circle cx="${x}" cy="${y}" r="3.5" fill="var(--accent)"/>`;
  }).join("");

  return `<svg class="sparkline" viewBox="0 0 ${width} ${height}">
    <path d="${area}" fill="var(--accent)" opacity="0.08"/>
    <path d="${path}" fill="none" stroke="var(--accent)" stroke-width="2.2"/>
    ${dots}
  </svg>`;
}

function renderWaveform(trace) {
  const values = trace.values || [];
  if (!values.length) return `<p class="empty-state">No waveform samples.</p>`;

  const width = 250;
  const height = 120;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;

  const path = values.map((value, index) => {
    const x = 12 + (index * (width - 24)) / Math.max(values.length - 1, 1);
    const y = height - 14 - ((value - min) / range) * (height - 28);
    return `${index === 0 ? "M" : "L"} ${x} ${y}`;
  }).join(" ");

  return `<svg class="waveform" viewBox="0 0 ${width} ${height}">
    <line x1="12" y1="${height / 2}" x2="${width - 12}" y2="${height / 2}" stroke="var(--border)" stroke-width="1"/>
    <path d="${path}" fill="none" stroke="var(--acc)" stroke-width="1.8"/>
  </svg>`;
}

function getVisibleSensors() {
  return state.sensorLayout.filter(isVisible);
}

function isVisible(sensor) {
  if (state.family !== "ALL" && sensor.measurement_family !== state.family) return false;

  if (state.isolatedSensorId) {
    const isolated = state.sensorLayout.find((item) => item.sensor_id === state.isolatedSensorId);
    if (isolated) {
      const sameNeighborhood =
        sensor.deck === isolated.deck
        && sensor.span === isolated.span
        && sensor.section === isolated.section;
      const isPair =
        sensor.sensor_id === isolated.sensor_id
        || sensor.sensor_id === isolated.homologous_sensor_id;
      if (!sameNeighborhood && !isPair) return false;
    }
  }

  return (
    passesFilter("deck", sensor.deck)
    && passesFilter("span", sensor.span)
    && passesFilter("side", sensor.side)
    && passesFilter("section", sensor.section)
    && passesFilter("axis_or_fibre", sensor.axis_or_fibre)
  );
}

function passesFilter(group, value) {
  const selected = state.filters[group];
  return selected.size === 0 || selected.has(value);
}

function reconcileSelection() {
  if (!state.selectedSensorId) return;
  const sensor = state.sensorLayout.find((item) => item.sensor_id === state.selectedSensorId);
  if (!sensor || !isVisible(sensor)) {
    state.selectedSensorId = null;
    state.selectedPreviewEventId = null;
    syncSelectionLight();
  }
}

function syncSelectionLight() {
  if (selectionLight?.parent) {
    selectionLight.parent.remove(selectionLight);
  }
  selectionLight = null;

  if (!state.selectedSensorId) return;
  const selectedObject = sensorObjs.get(state.selectedSensorId);
  if (!selectedObject) return;

  selectionLight = new THREE.PointLight(0x2e6ab4, 2.0, 14);
  selectionLight.position.set(0, 0, 0);
  selectedObject.group.add(selectionLight);
}

function applyWireframeMode() {
  const deckOpacity = state.wireframe ? 0.08 : null;
  const edgeOpacity = state.wireframe ? 0.85 : null;

  for (const entry of deckMaterials) {
    entry.material.transparent = true;
    entry.material.opacity = deckOpacity ?? entry.opacity;
  }

  for (const entry of deckEdgeMaterials) {
    entry.material.transparent = true;
    entry.material.opacity = edgeOpacity ?? entry.opacity;
  }

  if (minorGrid) {
    setHelperOpacity(minorGrid, state.wireframe ? 0.08 : 0.14);
  }
  if (majorGrid) {
    setHelperOpacity(majorGrid, state.wireframe ? 0.18 : 0.22);
  }

  renderer.setClearColor(state.wireframe ? 0xf8fafc : 0xf4f7fb, 1);

  for (const object of sensorObjs.values()) {
    if (object.group.userData.labelSprite) {
      object.group.userData.labelSprite.visible = state.wireframe;
    }
  }
}

function getRelatedSensors(sensor) {
  const family = sensor.measurement_family;
  return state.sensorLayout
    .filter((item) => {
      if (item.sensor_id === sensor.sensor_id) return false;
      if (state.family !== "ALL" && item.measurement_family !== state.family) return false;
      if (item.measurement_family !== family) return false;
      if (family === "ACC") {
        return (
          item.deck === sensor.deck
          && item.span === sensor.span
          && item.section === sensor.section
        );
      }
      return item.deck === sensor.deck && item.span === sensor.span;
    })
    .slice(0, 4);
}

function getMetric(sensorId) {
  return state.metricLookup.get(metricKey(state.selectedDataset, state.selectedMetric, sensorId));
}

function renderSegmented(container, values, active, onChange) {
  container.innerHTML = values
    .map(
      (value) => `<button type="button" class="${value === active ? "active" : ""}" data-segment-value="${value}">${value}</button>`
    )
    .join("");

  container.addEventListener("click", (event) => {
    const target = event.target.closest("[data-segment-value]");
    if (target) onChange(target.dataset.segmentValue);
  });
}

function fetchJson(path) {
  return fetch(path).then(async (response) => {
    if (!response.ok) throw new Error(`Failed to load ${path}: ${response.status}`);
    return response.json();
  });
}

function uniqueValues(values) {
  return [...new Set(values)].sort();
}

function labelForMetric(metricId) {
  const metric = state.manifest.metric_catalog.find((item) => item.metric_id === metricId);
  return metric ? metric.label : metricId;
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

function metricKey(dataset, metricId, sensorId) {
  return `${dataset}|${metricId}|${sensorId}`;
}

function capitalize(value) {
  return `${value.charAt(0).toUpperCase()}${value.slice(1)}`;
}

function humanizeSurface(surface) {
  return surface.replaceAll("_", " ");
}
