/**
 * NovaTurbo 3D Engine Viewer + Simulation Suite
 * Three.js ES Module viewer with thermal heatmap, airflow particles,
 * stress analysis, component controls, section cuts, and explode view.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { toggleDashboard } from './dashboard.js';

// ──── Color Maps ────
function thermalColorMap(t) {
    if (t < 0.25) { const s = t / 0.25; return new THREE.Color(0, s, 1); }
    if (t < 0.5)  { const s = (t - 0.25) / 0.25; return new THREE.Color(0, 1, 1 - s); }
    if (t < 0.75) { const s = (t - 0.5) / 0.25; return new THREE.Color(s, 1, 0); }
    const s = (t - 0.75) / 0.25; return new THREE.Color(1, 1 - s, 0);
}
function stressColorMap(t) {
    if (t < 0.33) { const s = t / 0.33; return new THREE.Color(0.1, 0.8 - s * 0.3, 0.1); }
    if (t < 0.66) { const s = (t - 0.33) / 0.33; return new THREE.Color(0.5 + s * 0.5, 0.5, 0); }
    const s = (t - 0.66) / 0.34; return new THREE.Color(1, 0.3 - s * 0.3, 0);
}
function flowColorMap(t) {
    if (t < 0.5) { const s = t / 0.5; return new THREE.Color(0.05, 0.1 + s * 0.5, 0.4 + s * 0.6); }
    const s = (t - 0.5) / 0.5; return new THREE.Color(s, 0.6 + s * 0.4, 1);
}
function flameColorMap(t) {
    if (t < 0.25) { const s = t / 0.25; return new THREE.Color(0.9 + 0.1 * s, 0.1, 0); }
    if (t < 0.5)  { const s = (t - 0.25) / 0.25; return new THREE.Color(1, 0.2 + 0.5 * s, 0); }
    if (t < 0.75) { const s = (t - 0.5) / 0.25; return new THREE.Color(1, 0.7 + 0.3 * s, 0.1 + 0.3 * s); }
    const s = (t - 0.75) / 0.25; return new THREE.Color(1, 1, 0.4 + 0.6 * s);
}

// State
const state = {
    scene: null,
    camera: null,
    renderer: null,
    controls: null,
    components: {},
    clippingPlane: null,
    wireframe: false,
    sectionActive: false,
    explodeActive: false,
    explodeFactor: 0,
    allVisible: true,
    totalVerts: 0,
    totalFaces: 0,
    engineCenter: new THREE.Vector3(),
    engineLength: 200,
    // Simulation
    simMode: null,
    simData: null,
    originalMaterials: {},
    flowParticles: null,
    flowArrows: [],
    flameParticles: null,
    flameVolumes: [],
    flameMeta: null,
    clock: new THREE.Clock(),
    // Lattice
    latticeVisible: false,
    latticeMeshes: {},
    currentVariation: 'v1_gyroid_standard',
};

// --- Init ---
function init() {
    setupScene();
    setupLights();
    setupGrid();
    loadComponents();
    loadEngineData();
    bindControls();
    animate();
}

function setupScene() {
    const canvas = document.getElementById('three-canvas');
    state.renderer = new THREE.WebGLRenderer({
        canvas,
        antialias: true,
        alpha: false,
        preserveDrawingBuffer: true
    });
    state.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    state.renderer.setClearColor(0x0a0e17, 1);
    state.renderer.setSize(canvas.parentElement.clientWidth, canvas.parentElement.clientHeight);
    state.renderer.localClippingEnabled = true;
    state.renderer.shadowMap.enabled = true;
    state.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    state.renderer.toneMappingExposure = 1.2;

    state.scene = new THREE.Scene();

    state.camera = new THREE.PerspectiveCamera(
        45,
        canvas.parentElement.clientWidth / canvas.parentElement.clientHeight,
        0.1,
        10000
    );
    state.camera.position.set(150, 100, 200);

    state.controls = new OrbitControls(state.camera, canvas);
    state.controls.enableDamping = true;
    state.controls.dampingFactor = 0.08;
    state.controls.rotateSpeed = 0.8;
    state.controls.target.set(0, 0, 100);

    state.clippingPlane = new THREE.Plane(new THREE.Vector3(-1, 0, 0), 200);

    window.addEventListener('resize', onResize);
    onResize();
}

function setupLights() {
    state.scene.add(new THREE.AmbientLight(0x404060, 0.6));
    state.scene.add(new THREE.HemisphereLight(0x88aacc, 0x223344, 0.5));

    const key = new THREE.DirectionalLight(0xffffff, 1.0);
    key.position.set(100, 150, 200);
    key.castShadow = true;
    state.scene.add(key);

    const fill = new THREE.DirectionalLight(0x6688aa, 0.4);
    fill.position.set(-100, 50, -100);
    state.scene.add(fill);

    const rim = new THREE.DirectionalLight(0x00E5FF, 0.2);
    rim.position.set(0, -50, -200);
    state.scene.add(rim);
}

function setupGrid() {
    const grid = new THREE.GridHelper(400, 40, 0x1e293b, 0x111827);
    grid.rotation.x = Math.PI / 2;
    grid.position.z = 100;
    state.scene.add(grid);
}

// --- Load STL Components ---
async function loadComponents() {
    const resp = await fetch('/api/components');
    const components = await resp.json();
    const loader = new STLLoader();
    let loaded = 0;

    const progress = document.getElementById('load-progress');
    progress.textContent = '0 / ' + components.length + ' components';

    for (const comp of components) {
        // Skip lattice meshes from auto-load (loaded on-demand via Lattice button)
        if (comp.id.includes('lattice')) {
            loaded++;
            continue;
        }
        try {
            const geometry = await new Promise((resolve, reject) => {
                loader.load(comp.file, resolve, undefined, reject);
            });

            geometry.computeVertexNormals();

            const material = new THREE.MeshPhysicalMaterial({
                color: new THREE.Color(comp.color),
                metalness: 0.7,
                roughness: 0.3,
                clearcoat: 0.1,
                clearcoatRoughness: 0.4,
                side: THREE.DoubleSide,
                clippingPlanes: [],
            });

            const mesh = new THREE.Mesh(geometry, material);
            mesh.castShadow = true;
            mesh.receiveShadow = true;

            state.scene.add(mesh);
            state.components[comp.id] = {
                mesh,
                data: comp,
                visible: true,
                originalPosition: mesh.position.clone()
            };
            state.originalMaterials[comp.id] = material.clone();

            state.totalVerts += geometry.attributes.position.count;
            state.totalFaces += geometry.index ? geometry.index.count / 3 : geometry.attributes.position.count / 3;

            loaded++;
            progress.textContent = loaded + ' / ' + components.length + ' components';

        } catch(e) {
            console.warn('Failed to load ' + comp.id, e);
            loaded++;
        }
    }

    // Compute center for camera target
    const box = new THREE.Box3();
    Object.values(state.components).forEach(c => box.expandByObject(c.mesh));
    box.getCenter(state.engineCenter);
    state.engineLength = box.max.z - box.min.z;

    state.controls.target.copy(state.engineCenter);
    state.camera.position.set(
        state.engineCenter.x + state.engineLength * 0.8,
        state.engineCenter.y + state.engineLength * 0.5,
        state.engineCenter.z + state.engineLength * 0.3
    );

    document.getElementById('stat-verts').textContent = formatNum(state.totalVerts);
    document.getElementById('stat-faces').textContent = formatNum(state.totalFaces);
    document.getElementById('status-left').textContent = 'Loaded ' + loaded + ' components';

    buildComponentList(components);

    setTimeout(() => {
        document.getElementById('loading-screen').classList.add('hidden');
    }, 300);
}

// --- Component List UI ---
function buildComponentList(components) {
    const list = document.getElementById('component-list');
    list.innerHTML = '';

    components.forEach(comp => {
        if (comp.id.includes('lattice')) return; // Lattice shown via button
        const item = document.createElement('div');
        item.className = 'comp-item';
        item.dataset.id = comp.id;
        item.innerHTML =
            '<div class="comp-swatch" style="background:' + comp.color + '"></div>' +
            '<span class="comp-label">' + comp.label + '</span>' +
            '<span class="comp-size">' + comp.size_kb.toFixed(0) + 'KB</span>' +
            '<svg class="comp-eye" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">' +
            '<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/>' +
            '<circle cx="12" cy="12" r="3"/></svg>';

        item.addEventListener('click', () => toggleComponent(comp.id, item));
        list.appendChild(item);
    });
}

function toggleComponent(id, element) {
    const comp = state.components[id];
    if (!comp) return;
    comp.visible = !comp.visible;
    comp.mesh.visible = comp.visible;
    element.classList.toggle('hidden-comp', !comp.visible);
}

// --- Engine Data Panel ---
async function loadEngineData() {
    try {
        const resp = await fetch('/api/engine');
        const data = await resp.json();
        renderEngineInfo(data);
    } catch(e) {
        document.getElementById('engine-info').innerHTML =
            '<div class="loading-placeholder">Could not load engine data</div>';
    }
}

function renderEngineInfo(data) {
    const el = document.getElementById('engine-info');
    let html = '';

    if (data.metrics && !data.metrics.error) {
        const m = data.metrics;
        html += '<div class="info-section"><h4>Geometry</h4>';
        html += infoRow('Engine', m.name, 'accent');
        html += infoRow('Length', m.total_length_mm + ' mm');
        html += infoRow('Max Diameter', m.max_diameter_mm + ' mm');
        html += infoRow('Total Mass', m.total_mass_g + ' g', 'accent');
        html += '</div>';

        html += '<div class="info-section"><h4>Specifications</h4>';
        html += infoRow('Pressure Ratio', m.specs.pressure_ratio);
        html += infoRow('RPM', formatNum(m.specs.rpm));
        html += infoRow('TIT', m.specs.tit_k + ' K (' + m.specs.tit_c + ' \u00B0C)', 'warning');
        html += infoRow('Shaft Dia', m.specs.shaft_diameter_mm + ' mm');
        html += '</div>';

        html += '<div class="info-section"><h4>Mass Breakdown</h4>';
        const total = Object.values(m.mass_breakdown).reduce((a, b) => a + b, 0);
        const colors = ['#4FC3F7','#81C784','#FF8A65','#FFD54F','#E57373','#B0BEC5','#9575CD'];
        let ci = 0;
        html += '<div class="mass-bar">';
        for (const [k, v] of Object.entries(m.mass_breakdown)) {
            const pct = (v / total * 100).toFixed(1);
            html += '<div class="mass-bar-segment" style="width:' + pct + '%;background:' + colors[ci % colors.length] + '" title="' + k + ': ' + v + 'g (' + pct + '%)"></div>';
            ci++;
        }
        html += '</div>';
        for (const [k, v] of Object.entries(m.mass_breakdown)) {
            html += infoRow(capitalize(k), v.toFixed(1) + ' g');
        }
        html += '</div>';
    }

    if (data.performance) {
        const p = data.performance;
        html += '<div class="info-section"><h4>AI Dataset</h4>';
        html += infoRow('Total Designs', formatNum(p.total_designs));
        html += infoRow('Valid', formatNum(p.valid_designs), 'success');
        html += infoRow('Thrust Range', p.thrust_range[0] + ' \u2013 ' + p.thrust_range[1] + ' N');
        html += infoRow('T/W Range', p.tw_range[0] + ' \u2013 ' + p.tw_range[1]);
        html += infoRow('Efficiency', p.efficiency_range[0] + ' \u2013 ' + p.efficiency_range[1] + '%');
        html += infoRow('Mean Thrust', p.mean_thrust + ' N', 'accent');
        html += infoRow('Mean T/W', p.mean_tw, 'accent');
        html += '</div>';
    }

    if (data.metrics && data.metrics.positions) {
        html += '<div class="info-section"><h4>Axial Positions</h4>';
        for (const [k, v] of Object.entries(data.metrics.positions)) {
            html += infoRow(capitalize(k), 'z = ' + v + ' mm');
        }
        html += '</div>';
    }

    el.innerHTML = html;
}

function infoRow(label, value, cls) {
    return '<div class="info-row"><span class="info-label">' + label +
           '</span><span class="info-value' + (cls ? ' ' + cls : '') + '">' +
           value + '</span></div>';
}

// --- Toolbar Controls ---
function bindControls() {
    document.getElementById('btn-reset-cam').addEventListener('click', resetCamera);
    document.getElementById('btn-wireframe').addEventListener('click', toggleWireframe);
    document.getElementById('btn-section').addEventListener('click', toggleSection);
    document.getElementById('btn-explode').addEventListener('click', toggleExplode);
    document.getElementById('btn-screenshot').addEventListener('click', takeScreenshot);
    document.getElementById('btn-toggle-all').addEventListener('click', toggleAll);

    // Simulation buttons
    document.getElementById('btn-sim-thermal').addEventListener('click', function() {
        if (state.simMode === 'thermal') { clearSimulation(); loadEngineData(); }
        else { activateThermalSim(); }
    });
    document.getElementById('btn-sim-flow').addEventListener('click', function() {
        if (state.simMode === 'flow') { clearSimulation(); loadEngineData(); }
        else { activateFlowSim(); }
    });
    document.getElementById('btn-sim-stress').addEventListener('click', function() {
        if (state.simMode === 'stress') { clearSimulation(); loadEngineData(); }
        else { activateStressSim(); }
    });
    document.getElementById('btn-sim-flame').addEventListener('click', function() {
        if (state.simMode === 'flame') { clearSimulation(); loadEngineData(); }
        else { activateFlameSim(); }
    });

    // Lattice toggle
    document.getElementById('btn-lattice').addEventListener('click', toggleLattice);

    // Dashboard toggle
    document.getElementById('btn-dashboard').addEventListener('click', function() {
        toggleDashboard();
    });

    document.getElementById('slider-opacity').addEventListener('input', function() {
        const v = parseFloat(this.value);
        document.getElementById('opacity-val').textContent = Math.round(v * 100) + '%';
        Object.values(state.components).forEach(c => {
            c.mesh.material.opacity = v;
            c.mesh.material.transparent = v < 1;
            c.mesh.material.needsUpdate = true;
        });
    });

    document.getElementById('slider-section').addEventListener('input', function() {
        const v = parseInt(this.value);
        document.getElementById('section-val').textContent = v + '%';
        const range = state.engineLength * 1.2;
        state.clippingPlane.constant = (v / 100) * range - range * 0.1 + state.engineCenter.x;
    });

    document.getElementById('slider-explode').addEventListener('input', function() {
        const v = parseInt(this.value);
        document.getElementById('explode-val').textContent = v + '%';
        state.explodeFactor = v / 100;
        applyExplode();
    });
}

function resetCamera() {
    state.controls.target.copy(state.engineCenter);
    state.camera.position.set(
        state.engineCenter.x + state.engineLength * 0.8,
        state.engineCenter.y + state.engineLength * 0.5,
        state.engineCenter.z + state.engineLength * 0.3
    );
}

function toggleWireframe() {
    state.wireframe = !state.wireframe;
    document.getElementById('btn-wireframe').classList.toggle('active', state.wireframe);
    Object.values(state.components).forEach(c => {
        c.mesh.material.wireframe = state.wireframe;
    });
}

function toggleSection() {
    state.sectionActive = !state.sectionActive;
    document.getElementById('btn-section').classList.toggle('active', state.sectionActive);
    document.getElementById('section-controls').style.display = state.sectionActive ? 'flex' : 'none';

    const planes = state.sectionActive ? [state.clippingPlane] : [];
    Object.values(state.components).forEach(c => {
        c.mesh.material.clippingPlanes = planes;
        c.mesh.material.needsUpdate = true;
    });
}

function toggleExplode() {
    state.explodeActive = !state.explodeActive;
    document.getElementById('btn-explode').classList.toggle('active', state.explodeActive);
    document.getElementById('explode-controls').style.display = state.explodeActive ? 'flex' : 'none';
    if (!state.explodeActive) {
        state.explodeFactor = 0;
        document.getElementById('slider-explode').value = 0;
        document.getElementById('explode-val').textContent = '0%';
        applyExplode();
    }
}

function applyExplode() {
    const order = ['inlet','compressor','combustor','turbine','nozzle','shaft','casing','transitions'];
    order.forEach((id, i) => {
        const comp = state.components[id];
        if (!comp) return;
        const offset = (i - order.length / 2) * state.explodeFactor * 40;
        comp.mesh.position.z = comp.originalPosition.z + offset;
    });
}

function toggleAll() {
    state.allVisible = !state.allVisible;
    document.getElementById('btn-toggle-all').textContent = state.allVisible ? 'Hide All' : 'Show All';

    Object.keys(state.components).forEach(id => {
        state.components[id].visible = state.allVisible;
        state.components[id].mesh.visible = state.allVisible;
    });

    document.querySelectorAll('.comp-item').forEach(el => {
        el.classList.toggle('hidden-comp', !state.allVisible);
    });
}

function takeScreenshot() {
    state.renderer.render(state.scene, state.camera);
    const link = document.createElement('a');
    link.download = 'novaturbo_engine.png';
    link.href = state.renderer.domElement.toDataURL('image/png');
    link.click();
    document.getElementById('status-left').textContent = 'Screenshot saved';
}

// --- Animation Loop ---
let frameCount = 0;
let lastTime = performance.now();

function animate() {
    requestAnimationFrame(animate);
    state.controls.update();

    // Animate flow particles
    if (state.simMode === 'flow' && state.flowParticles) {
        var positions = state.flowParticles.geometry.attributes.position;
        var velocities = state.flowParticles.geometry.attributes.velocity;
        var dt = state.clock.getDelta() * 15;
        for (var i = 0; i < positions.count; i++) {
            var z = positions.getZ(i);
            z += velocities.getZ(i) * dt * 0.05;
            if (z > state.engineLength + 30) z = -10 + Math.random() * 10;
            if (z < -20) z = state.engineLength + Math.random() * 10;
            positions.setZ(i, z);
        }
        positions.needsUpdate = true;
    }

    // Animate flame particles
    if (state.simMode === 'flame' && state.flameParticles && state.flameMeta) {
        var flamePos = state.flameParticles.geometry.attributes.position;
        var flameVel = state.flameParticles.geometry.attributes.velocity;
        var flameRegion = state.flameParticles.geometry.attributes.region;
        var dtf = state.clock.getDelta() * 25;
        var cb = state.flameMeta.combustor;
        var nb = state.flameMeta.nozzle;
        var plumeEnd = nb.zMax + state.flameMeta.plumeLength;

        for (var fi = 0; fi < flamePos.count; fi++) {
            var x = flamePos.getX(fi);
            var y = flamePos.getY(fi);
            var z = flamePos.getZ(fi);
            var vx = flameVel.getX(fi);
            var vy = flameVel.getY(fi);
            var vz = flameVel.getZ(fi);
            var reg = flameRegion.getX(fi);

            // Spiral turbulence for a fume-like motion
            var swirl = reg < 0.5 ? 0.0016 : 0.0009;
            vx += -y * swirl * dtf + (Math.random() - 0.5) * 0.02;
            vy +=  x * swirl * dtf + (Math.random() - 0.5) * 0.02;

            x += vx * dtf * 0.06;
            y += vy * dtf * 0.06;
            z += vz * dtf * 0.07;

            var r = Math.sqrt(x * x + y * y);
            var maxR = reg < 0.5 ? cb.radius : nb.radius + 6.0;
            var zStart = reg < 0.5 ? cb.zMin : nb.zMin;
            var zEnd = reg < 0.5 ? cb.zMax : plumeEnd;

            // Respawn when particle exits the flame envelope
            if (z > zEnd || z < zStart || r > maxR) {
                var theta = Math.random() * Math.PI * 2;
                var rr = Math.sqrt(Math.random()) * (reg < 0.5 ? cb.radius * 0.8 : nb.radius * 0.7);
                x = Math.cos(theta) * rr;
                y = Math.sin(theta) * rr;
                z = reg < 0.5
                    ? cb.zMin + Math.random() * (cb.zMax - cb.zMin) * 0.65
                    : nb.zMin + Math.random() * Math.max((nb.zMax - nb.zMin) * 0.25, 3.0);
                vx = (Math.random() - 0.5) * 0.9;
                vy = (Math.random() - 0.5) * 0.9;
                vz = reg < 0.5 ? 3.0 + Math.random() * 10.0 : 8.0 + Math.random() * 16.0;
            }

            flamePos.setXYZ(fi, x, y, z);
            flameVel.setXYZ(fi, vx, vy, vz);
        }
        flamePos.needsUpdate = true;
        flameVel.needsUpdate = true;

        // Subtle breathing of volumetric shells
        for (var fv = 0; fv < state.flameVolumes.length; fv++) {
            var shell = state.flameVolumes[fv];
            var pulse = 1.0 + 0.02 * Math.sin(performance.now() * 0.006 + fv * 1.2);
            shell.scale.set(pulse, pulse, 1.0);
            shell.material.opacity = 0.12 + 0.03 * Math.sin(performance.now() * 0.01 + fv);
        }
    }

    state.renderer.render(state.scene, state.camera);

    frameCount++;
    const now = performance.now();
    if (now - lastTime >= 1000) {
        document.getElementById('stat-fps').textContent = frameCount;
        frameCount = 0;
        lastTime = now;
    }
}

function onResize() {
    const vp = document.getElementById('viewport');
    const w = vp.clientWidth;
    const h = vp.clientHeight;
    state.camera.aspect = w / h;
    state.camera.updateProjectionMatrix();
    state.renderer.setSize(w, h);
}

// --- Helpers ---
function formatNum(n) { return n.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ','); }
function capitalize(s) { return s.charAt(0).toUpperCase() + s.slice(1); }

function interpolateArray(xArr, yArr, x) {
    if (x <= xArr[0]) return yArr[0];
    if (x >= xArr[xArr.length - 1]) return yArr[yArr.length - 1];
    for (var i = 0; i < xArr.length - 1; i++) {
        if (x >= xArr[i] && x <= xArr[i + 1]) {
            var t = (x - xArr[i]) / (xArr[i + 1] - xArr[i]);
            return yArr[i] + t * (yArr[i + 1] - yArr[i]);
        }
    }
    return yArr[yArr.length - 1];
}

// ──── Simulation Engine ────
async function loadSimulation() {
    if (state.simData) return state.simData;
    document.getElementById('status-left').textContent = 'Running simulation...';
    try {
        var resp = await fetch('/api/simulation');
        state.simData = await resp.json();
        document.getElementById('status-left').textContent = 'Simulation complete';
        return state.simData;
    } catch(e) {
        document.getElementById('status-left').textContent = 'Simulation failed';
        return null;
    }
}

function clearSimulation() {
    for (var id in state.components) {
        var comp = state.components[id];
        if (state.originalMaterials[id]) {
            comp.mesh.material = state.originalMaterials[id].clone();
            if (state.sectionActive) comp.mesh.material.clippingPlanes = [state.clippingPlane];
        }
        if (comp.mesh.geometry.attributes.color) {
            comp.mesh.geometry.deleteAttribute('color');
        }
    }
    if (state.flowParticles) {
        state.scene.remove(state.flowParticles);
        state.flowParticles = null;
    }
    if (state.flameParticles) {
        state.scene.remove(state.flameParticles);
        if (state.flameParticles.geometry) state.flameParticles.geometry.dispose();
        if (state.flameParticles.material) state.flameParticles.material.dispose();
        state.flameParticles = null;
    }
    state.flameVolumes.forEach(function(m) {
        state.scene.remove(m);
        if (m.geometry) m.geometry.dispose();
        if (m.material) m.material.dispose();
    });
    state.flameVolumes = [];
    state.flameMeta = null;
    state.flowArrows.forEach(function(a) { state.scene.remove(a); });
    state.flowArrows = [];
    document.getElementById('sim-legend').style.display = 'none';
    document.getElementById('sim-badge').style.display = 'none';
    ['btn-sim-thermal', 'btn-sim-flow', 'btn-sim-stress', 'btn-sim-flame'].forEach(function(btnId) {
        document.getElementById(btnId).classList.remove('active', 'active-thermal', 'active-flow', 'active-stress', 'active-flame');
    });
    state.simMode = null;
}

async function activateThermalSim() {
    var data = await loadSimulation();
    if (!data) return;
    clearSimulation();
    state.simMode = 'thermal';
    document.getElementById('btn-sim-thermal').classList.add('active-thermal');

    var thermal = data.thermal;
    var tMin = thermal.t_min;
    var tMax = thermal.t_max;

    for (var compId in state.components) {
        var comp = state.components[compId];
        var tempData = thermal.component_temps[compId];
        if (!tempData) continue;

        var geo = comp.mesh.geometry;
        var pos = geo.attributes.position;
        var colors = new Float32Array(pos.count * 3);
        var zOffset = tempData.z_offset || 0;

        for (var i = 0; i < pos.count; i++) {
            var z = pos.getZ(i);
            var localZ = z - zOffset;
            var temp = interpolateArray(tempData.z, tempData.wall_temp, localZ);
            var t = Math.max(0, Math.min(1, (temp - tMin) / (tMax - tMin)));
            var color = thermalColorMap(t);
            colors[i * 3] = color.r;
            colors[i * 3 + 1] = color.g;
            colors[i * 3 + 2] = color.b;
        }

        geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        comp.mesh.material = new THREE.MeshPhysicalMaterial({
            vertexColors: true, metalness: 0.3, roughness: 0.5,
            side: THREE.DoubleSide,
            clippingPlanes: state.sectionActive ? [state.clippingPlane] : [],
        });
    }

    showLegend('Temperature (K)', tMin, tMax, thermalColorMap);
    showBadge('THERMAL ANALYSIS', '#ef4444');
    renderSimInfo(data, 'thermal');
}

async function activateFlowSim() {
    var data = await loadSimulation();
    if (!data) return;
    clearSimulation();
    state.simMode = 'flow';
    document.getElementById('btn-sim-flow').classList.add('active-flow');

    var flow = data.flow;

    // Make engine semi-transparent
    for (var id in state.components) {
        var comp = state.components[id];
        comp.mesh.material = new THREE.MeshPhysicalMaterial({
            color: comp.data.color, metalness: 0.5, roughness: 0.4,
            transparent: true, opacity: 0.18, side: THREE.DoubleSide,
            clippingPlanes: state.sectionActive ? [state.clippingPlane] : [],
        });
    }

    // Create streamline tubes (smooth curves through flow points)
    for (var s = 0; s < flow.streamlines.length; s++) {
        var pts = flow.streamlines[s].points;
        if (pts.length < 3) continue;

        // Build curve points
        var curvePoints = [];
        for (var p = 0; p < pts.length; p++) {
            curvePoints.push(new THREE.Vector3(pts[p].x, pts[p].y, pts[p].z));
        }
        var curve = new THREE.CatmullRomCurve3(curvePoints);

        // Create tube with velocity-based color
        var tubePts = 80;
        var tubeGeo = new THREE.TubeGeometry(curve, tubePts, 1.2, 6, false);
        var colors = new Float32Array(tubeGeo.attributes.position.count * 3);
        for (var vi = 0; vi < tubeGeo.attributes.position.count; vi++) {
            var z = tubeGeo.attributes.position.getZ(vi);
            // Map z to speed
            var zNorm = (z - (pts[0].z || 0)) / (state.engineLength + 1);
            zNorm = Math.max(0, Math.min(1, zNorm));
            // Find closest point for speed
            var idx = Math.min(Math.floor(zNorm * (pts.length - 1)), pts.length - 1);
            var speed = pts[idx].speed || 0;
            var t = Math.max(0, Math.min(1, (speed - flow.v_min) / (flow.v_max - flow.v_min + 0.1)));
            var c = flowColorMap(t);
            colors[vi * 3] = c.r;
            colors[vi * 3 + 1] = c.g;
            colors[vi * 3 + 2] = c.b;
        }
        tubeGeo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        var tubeMat = new THREE.MeshPhysicalMaterial({
            vertexColors: true, metalness: 0.6, roughness: 0.3,
            transparent: true, opacity: 0.85, side: THREE.DoubleSide,
        });
        var tubeMesh = new THREE.Mesh(tubeGeo, tubeMat);
        state.scene.add(tubeMesh);
        state.flowArrows.push(tubeMesh);
    }

    // Animated particles along streamlines
    var allPoints = [];
    var allColors = [];
    var allVelocities = [];

    for (var s = 0; s < flow.streamlines.length; s++) {
        var pts = flow.streamlines[s].points;
        for (var p = 0; p < pts.length; p++) {
            var pt = pts[p];
            // Multiple particles per point, spread radially
            for (var j = 0; j < 5; j++) {
                var angle = Math.random() * Math.PI * 2;
                var spread = 2 + Math.random() * 5;
                allPoints.push(
                    pt.x + Math.cos(angle) * spread,
                    pt.y + Math.sin(angle) * spread,
                    pt.z + (Math.random() - 0.5) * 3
                );
                var t = Math.max(0, Math.min(1, (pt.speed - flow.v_min) / (flow.v_max - flow.v_min + 0.1)));
                var c = flowColorMap(t);
                allColors.push(c.r, c.g, c.b);
                allVelocities.push(pt.vx * 0.3, pt.vy * 0.3, pt.vz);
            }
        }
    }

    var particleGeo = new THREE.BufferGeometry();
    particleGeo.setAttribute('position', new THREE.Float32BufferAttribute(allPoints, 3));
    particleGeo.setAttribute('color', new THREE.Float32BufferAttribute(allColors, 3));
    particleGeo.setAttribute('velocity', new THREE.Float32BufferAttribute(allVelocities, 3));

    var particleMat = new THREE.PointsMaterial({
        size: 1.8, vertexColors: true, transparent: true, opacity: 0.7,
        sizeAttenuation: true, blending: THREE.AdditiveBlending, depthWrite: false,
    });

    state.flowParticles = new THREE.Points(particleGeo, particleMat);
    state.scene.add(state.flowParticles);

    // Station labels (text sprites at component boundaries)
    var stations = data.cycle.stations;
    var stationColors = { 'S0': '#94a3b8', 'S1': '#4FC3F7', 'S2': '#81C784', 'S3': '#FF8A65', 'S4': '#FFD54F', 'S5': '#E57373' };
    for (var key in stations) {
        var st = stations[key];
        if (st.z_mm !== undefined) {
            var label = createTextSprite(
                st.name + '\n' + st.T_total_K + 'K | ' + st.P_total_kPa + 'kPa',
                stationColors[key] || '#ffffff'
            );
            label.position.set(0, 70, st.z_mm || 0);
            state.scene.add(label);
            state.flowArrows.push(label);
        }
    }

    // Flow direction arrows (larger, at key stations)
    var arrowPositions = [
        { z: 15, r: 0, label: 'INTAKE' },
        { z: 70, r: 0, label: 'COMPRESS' },
        { z: 130, r: 0, label: 'BURN' },
        { z: 180, r: 0, label: 'EXPAND' },
        { z: 215, r: 0, label: 'EXHAUST' }
    ];
    for (var a = 0; a < arrowPositions.length; a++) {
        var ap = arrowPositions[a];
        var arrow = new THREE.ArrowHelper(
            new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, -55, ap.z),
            18, 0x3b82f6, 6, 4
        );
        state.scene.add(arrow);
        state.flowArrows.push(arrow);
    }

    showLegend('Velocity (m/s)', flow.v_min, flow.v_max, flowColorMap);
    showBadge('AIRFLOW ANALYSIS', '#3b82f6');
    renderSimInfo(data, 'flow');
}

async function activateStressSim() {
    var data = await loadSimulation();
    if (!data) return;
    clearSimulation();
    state.simMode = 'stress';
    document.getElementById('btn-sim-stress').classList.add('active-stress');

    var stress = data.stress;
    var sMin = stress.s_min;
    var sMax = stress.s_max;

    for (var compId in state.components) {
        var comp = state.components[compId];
        var stressData = stress.component_stress[compId];
        if (!stressData) continue;

        var geo = comp.mesh.geometry;
        var pos = geo.attributes.position;
        var colors = new Float32Array(pos.count * 3);
        var zOffset = stressData.z_offset || 0;

        for (var i = 0; i < pos.count; i++) {
            var z = pos.getZ(i);
            var localZ = z - zOffset;
            var sv = interpolateArray(stressData.z, stressData.total_MPa, localZ);
            var t = Math.max(0, Math.min(1, (sv - sMin) / (sMax - sMin + 0.1)));
            var color = stressColorMap(t);
            colors[i * 3] = color.r;
            colors[i * 3 + 1] = color.g;
            colors[i * 3 + 2] = color.b;
        }

        geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        comp.mesh.material = new THREE.MeshPhysicalMaterial({
            vertexColors: true, metalness: 0.3, roughness: 0.5,
            side: THREE.DoubleSide,
            clippingPlanes: state.sectionActive ? [state.clippingPlane] : [],
        });
    }

    showLegend('Stress (MPa)', sMin, sMax, stressColorMap);
    showBadge('STRESS ANALYSIS', '#f59e0b');
    renderSimInfo(data, 'stress');
}

function getComponentBounds(componentId) {
    var comp = state.components[componentId];
    if (!comp || !comp.mesh) return null;
    var box = new THREE.Box3().setFromObject(comp.mesh);
    return {
        zMin: box.min.z,
        zMax: box.max.z,
        radius: Math.max(box.max.x - box.min.x, box.max.y - box.min.y) * 0.5
    };
}

async function activateFlameSim() {
    var data = await loadSimulation();
    if (!data) return;
    clearSimulation();
    state.simMode = 'flame';
    document.getElementById('btn-sim-flame').classList.add('active-flame');

    // Fade solids to reveal volumetric flame core
    for (var id in state.components) {
        var comp = state.components[id];
        var op = (id === 'combustor' || id === 'nozzle') ? 0.12 : 0.22;
        comp.mesh.material = new THREE.MeshPhysicalMaterial({
            color: comp.data.color, metalness: 0.4, roughness: 0.45,
            transparent: true, opacity: op, side: THREE.DoubleSide,
            clippingPlanes: state.sectionActive ? [state.clippingPlane] : [],
        });
    }

    var cb = getComponentBounds('combustor');
    var nb = getComponentBounds('nozzle');
    if (!cb || !nb) {
        document.getElementById('status-left').textContent = 'Flame simulation unavailable: missing combustor/nozzle mesh';
        return;
    }

    state.flameMeta = {
        combustor: { zMin: cb.zMin, zMax: cb.zMax, radius: Math.max(8, cb.radius * 0.78) },
        nozzle: { zMin: nb.zMin, zMax: nb.zMax, radius: Math.max(4, nb.radius * 0.68) },
        plumeLength: 48.0
    };

    var allPoints = [];
    var allColors = [];
    var allVelocities = [];
    var allRegions = [];

    function pushParticle(region) {
        var meta = region === 0 ? state.flameMeta.combustor : state.flameMeta.nozzle;
        var theta = Math.random() * Math.PI * 2;
        var r = Math.sqrt(Math.random()) * meta.radius;
        var x = Math.cos(theta) * r;
        var y = Math.sin(theta) * r;
        var z = region === 0
            ? meta.zMin + Math.random() * (meta.zMax - meta.zMin) * 0.75
            : meta.zMin + Math.random() * Math.max((meta.zMax - meta.zMin) * 0.35, 4.0);
        var t = region === 0 ? Math.random() * 0.75 + 0.2 : Math.random() * 0.55 + 0.15;
        var c = flameColorMap(Math.min(1, t));

        allPoints.push(x, y, z);
        allColors.push(c.r, c.g, c.b);
        allVelocities.push(
            (Math.random() - 0.5) * (region === 0 ? 0.8 : 1.1),
            (Math.random() - 0.5) * (region === 0 ? 0.8 : 1.1),
            region === 0 ? (3.0 + Math.random() * 10.0) : (8.0 + Math.random() * 16.0)
        );
        allRegions.push(region);
    }

    for (var i = 0; i < 2200; i++) pushParticle(0); // combustor flame body
    for (var j = 0; j < 1300; j++) pushParticle(1); // nozzle plume

    var flameGeo = new THREE.BufferGeometry();
    flameGeo.setAttribute('position', new THREE.Float32BufferAttribute(allPoints, 3));
    flameGeo.setAttribute('color', new THREE.Float32BufferAttribute(allColors, 3));
    flameGeo.setAttribute('velocity', new THREE.Float32BufferAttribute(allVelocities, 3));
    flameGeo.setAttribute('region', new THREE.Float32BufferAttribute(allRegions, 1));

    var flameMat = new THREE.PointsMaterial({
        size: 2.8, vertexColors: true, transparent: true, opacity: 0.72,
        sizeAttenuation: true, blending: THREE.AdditiveBlending, depthWrite: false,
    });

    state.flameParticles = new THREE.Points(flameGeo, flameMat);
    state.scene.add(state.flameParticles);

    // Volumetric shells to emulate fume-like combustion volume
    var coreShellGeo = new THREE.CylinderGeometry(
        state.flameMeta.combustor.radius * 0.95,
        state.flameMeta.combustor.radius * 0.62,
        state.flameMeta.combustor.zMax - state.flameMeta.combustor.zMin,
        24, 1, true
    );
    coreShellGeo.rotateX(Math.PI / 2);
    coreShellGeo.translate(0, 0, (state.flameMeta.combustor.zMin + state.flameMeta.combustor.zMax) * 0.5);
    var coreShell = new THREE.Mesh(coreShellGeo, new THREE.MeshBasicMaterial({
        color: 0xff5a00, transparent: true, opacity: 0.12, side: THREE.DoubleSide,
        blending: THREE.AdditiveBlending, depthWrite: false
    }));
    state.scene.add(coreShell);
    state.flameVolumes.push(coreShell);

    var plumeGeo = new THREE.ConeGeometry(state.flameMeta.nozzle.radius * 0.72, state.flameMeta.plumeLength, 20, 1, true);
    plumeGeo.rotateX(Math.PI / 2);
    plumeGeo.translate(0, 0, state.flameMeta.nozzle.zMax + state.flameMeta.plumeLength * 0.5);
    var plume = new THREE.Mesh(plumeGeo, new THREE.MeshBasicMaterial({
        color: 0xff8a33, transparent: true, opacity: 0.1, side: THREE.DoubleSide,
        blending: THREE.AdditiveBlending, depthWrite: false
    }));
    state.scene.add(plume);
    state.flameVolumes.push(plume);

    showLegend('Flame Temp (K)', 900, 2200, flameColorMap);
    showBadge('COMBUSTION FLAME', '#ff6a00');
    renderSimInfo(data, 'flame');
}

function createTextSprite(text, color) {
    var canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 128;
    var ctx = canvas.getContext('2d');
    ctx.font = 'bold 20px Inter, sans-serif';
    ctx.fillStyle = color || '#ffffff';
    ctx.textAlign = 'center';
    var lines = text.split('\n');
    for (var i = 0; i < lines.length; i++) {
        ctx.fillText(lines[i], 128, 40 + i * 28);
    }
    var texture = new THREE.CanvasTexture(canvas);
    var mat = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false });
    var sprite = new THREE.Sprite(mat);
    sprite.scale.set(40, 20, 1);
    return sprite;
}

function showLegend(title, min, max, colorMapFn) {
    document.getElementById('sim-legend').style.display = 'block';
    document.getElementById('legend-title').textContent = title;
    document.getElementById('legend-max').textContent = Math.round(max);
    document.getElementById('legend-mid').textContent = Math.round((min + max) / 2);
    document.getElementById('legend-min').textContent = Math.round(min);
    var canvas = document.getElementById('legend-canvas');
    var ctx = canvas.getContext('2d');
    for (var y = 0; y < canvas.height; y++) {
        var t = 1 - y / canvas.height;
        var color = colorMapFn(t);
        ctx.fillStyle = 'rgb(' + Math.round(color.r * 255) + ',' + Math.round(color.g * 255) + ',' + Math.round(color.b * 255) + ')';
        ctx.fillRect(0, y, canvas.width, 1);
    }
}

function showBadge(text, color) {
    var badge = document.getElementById('sim-badge');
    badge.style.display = 'flex';
    document.getElementById('sim-badge-text').textContent = text;
    document.getElementById('sim-badge-icon').style.background = color;
    badge.style.color = color;
    badge.style.borderColor = color;
}

function renderSimInfo(data, mode) {
    var el = document.getElementById('engine-info');
    var html = '';
    var c = data.cycle;

    // Status header with overall health
    if (mode === 'thermal') {
        var allSafe = true;
        for (var comp in data.thermal.material_limits) {
            var temps = data.thermal.component_temps[comp];
            if (!temps) continue;
            var maxWall = Math.max.apply(null, temps.wall_temp);
            if (maxWall >= data.thermal.material_limits[comp]) allSafe = false;
        }
        html += '<div class="info-section" style="text-align:center;padding:12px;">';
        html += '<div style="font-size:28px;margin-bottom:4px;">' + (allSafe ? '✅' : '⚠️') + '</div>';
        html += '<div style="font-size:13px;font-weight:600;color:' + (allSafe ? 'var(--success)' : 'var(--warning)') + ';">';
        html += allSafe ? 'ALL TEMPERATURES WITHIN LIMITS' : 'THERMAL WARNING DETECTED';
        html += '</div></div>';
    }

    if (mode === 'stress') {
        var allSafe = true;
        for (var sc in data.stress.yield_margins) {
            if (data.stress.yield_margins[sc] <= 0) allSafe = false;
        }
        html += '<div class="info-section" style="text-align:center;padding:12px;">';
        html += '<div style="font-size:28px;margin-bottom:4px;">' + (allSafe ? '✅' : '🔴') + '</div>';
        html += '<div style="font-size:13px;font-weight:600;color:' + (allSafe ? 'var(--success)' : 'var(--danger)') + ';">';
        html += allSafe ? 'ALL STRESSES BELOW YIELD' : 'STRUCTURAL FAILURE RISK';
        html += '</div></div>';
    }

    if (mode === 'flow') {
        html += '<div class="info-section" style="text-align:center;padding:12px;">';
        html += '<div style="font-size:28px;margin-bottom:4px;">🌊</div>';
        html += '<div style="font-size:13px;font-weight:600;color:var(--accent);">AIRFLOW ANALYSIS</div>';
        html += '<div style="font-size:11px;color:var(--text-secondary);margin-top:4px;">Streamlines show velocity-colored flow paths through the engine</div>';
        html += '</div>';
    }

    if (mode === 'flame') {
        html += '<div class="info-section" style="text-align:center;padding:12px;">';
        html += '<div style="font-size:28px;margin-bottom:4px;">🔥</div>';
        html += '<div style="font-size:13px;font-weight:600;color:#ff6a00;">COMBUSTION FLAME VISUALIZATION</div>';
        html += '<div style="font-size:11px;color:var(--text-secondary);margin-top:4px;">Particle fume field shows hot core + nozzle plume dynamics</div>';
        html += '</div>';
    }

    html += '<div class="info-section"><h4>Cycle Performance</h4>';
    html += infoRow('Thrust', c.thrust_N + ' N (' + c.thrust_kgf + ' kgf)', 'accent');
    html += infoRow('Exhaust Vel.', c.exhaust_velocity + ' m/s');
    html += infoRow('Exhaust Temp', Math.round(c.exhaust_temp_K) + ' K', 'warning');
    html += infoRow('Fuel Flow', c.fuel_flow_g_hr + ' g/hr');
    html += infoRow('TSFC', c.tsfc_g_kNs + ' g/kN\u00B7s');
    html += infoRow('Thermal Eff.', c.thermal_efficiency_pct + '%', 'success');
    html += infoRow('Comp. Power', c.compressor_power_kW + ' kW');
    html += infoRow('Turb. Power', c.turbine_power_kW + ' kW');
    html += '</div>';

    html += '<div class="info-section"><h4>Station Analysis</h4>';
    var stationLabels = {
        'S0': '🌬️ Ambient', 'S1': '📥 Inlet Exit',
        'S2': '🔄 Compressor Exit', 'S3': '🔥 Combustor Exit',
        'S4': '⚙️ Turbine Exit', 'S5': '💨 Nozzle Exit'
    };
    for (var key in c.stations) {
        var st = c.stations[key];
        var label = stationLabels[key] || st.name;
        html += infoRow(label, st.T_total_K + ' K / ' + st.P_total_kPa + ' kPa');
    }
    html += '</div>';

    if (mode === 'thermal') {
        html += '<div class="info-section"><h4>🌡️ Thermal Limits</h4>';
        html += '<div style="font-size:10px;color:var(--text-muted);margin-bottom:8px;">Max wall temp vs material limit. Green = safe, Yellow = caution.</div>';
        for (var comp in data.thermal.material_limits) {
            var temps = data.thermal.component_temps[comp];
            if (!temps) continue;
            var maxWall = Math.round(Math.max.apply(null, temps.wall_temp));
            var limit = data.thermal.material_limits[comp];
            var margin = limit - maxWall;
            var pct = Math.round(maxWall / limit * 100);
            var status = maxWall < limit * 0.85 ? 'success' : (maxWall < limit ? 'warning' : 'danger');
            html += '<div class="info-row">';
            html += '<span class="info-label">' + capitalize(comp) + '</span>';
            html += '<span class="info-value ' + status + '">' + maxWall + ' / ' + limit + ' K</span>';
            html += '</div>';
            // Progress bar showing how close to limit
            html += '<div style="height:4px;background:var(--bg-hover);border-radius:2px;margin:-4px 0 6px 0;">';
            html += '<div style="height:100%;width:' + Math.min(pct, 100) + '%;border-radius:2px;background:';
            html += status === 'success' ? 'var(--success)' : (status === 'warning' ? 'var(--warning)' : 'var(--danger)');
            html += ';"></div></div>';
        }
        html += '</div>';
    }

    if (mode === 'flow') {
        html += '<div class="info-section"><h4>🌊 Flow Stages</h4>';
        html += '<div style="font-size:10px;color:var(--text-muted);margin-bottom:8px;">Pressure ratio and temperature change per component</div>';
        for (var fc in data.flow.component_flow) {
            var fd = data.flow.component_flow[fc];
            html += '<div class="info-row">';
            html += '<span class="info-label">' + capitalize(fc) + '</span>';
            html += '<span class="info-value">PR=' + fd.pressure_ratio + '</span>';
            html += '</div>';
            html += '<div style="font-size:10px;color:var(--text-secondary);margin:-2px 0 6px 12px;">';
            html += fd.inlet_temp_K + 'K → ' + fd.outlet_temp_K + 'K (Δ' + (fd.outlet_temp_K - fd.inlet_temp_K > 0 ? '+' : '') + (fd.outlet_temp_K - fd.inlet_temp_K) + 'K)';
            html += '</div>';
        }
        html += '</div>';

        html += '<div class="info-section"><h4>📊 Flow Legend</h4>';
        html += '<div style="font-size:10px;color:var(--text-muted);margin-bottom:6px;">';
        html += '• <span style="color:#0a1a4a;">Dark blue</span> = Slow flow (intake)<br>';
        html += '• <span style="color:#1a9fff;">Cyan</span> = Medium flow<br>';
        html += '• <span style="color:#ffffff;">White</span> = Fast flow (exhaust)<br>';
        html += '• Tubes show streamline paths<br>';
        html += '• Particles animate flow direction';
        html += '</div></div>';
    }

    if (mode === 'flame') {
        var s3 = null, s5 = null;
        if (c.stations) {
            s3 = c.stations['4_combustor_exit'] || c.stations['S3'] || null;
            s5 = c.stations['6_nozzle_exit'] || c.stations['S5'] || null;
        }
        html += '<div class="info-section"><h4>🔥 Flame Core</h4>';
        html += infoRow('Combustor Core', s3 ? s3.T_total_K + ' K' : 'N/A', 'warning');
        html += infoRow('Nozzle Exit', s5 ? s5.T_total_K + ' K' : 'N/A');
        html += infoRow('Fuel Flow', c.fuel_flow_g_hr + ' g/hr');
        html += infoRow('Visual Mode', 'Fume-style particle plume', 'accent');
        html += '</div>';

        html += '<div class="info-section"><h4>📊 Flame Legend</h4>';
        html += '<div style="font-size:10px;color:var(--text-muted);">';
        html += '• <span style="color:#ff3b00;">Red/Orange</span> = Hot reaction zone<br>';
        html += '• <span style="color:#ffd166;">Yellow/White</span> = Peak combustion<br>';
        html += '• Core shell = combustor volume<br>';
        html += '• Cone plume = exhaust flame tail';
        html += '</div></div>';
    }

    if (mode === 'stress') {
        html += '<div class="info-section"><h4>🛡️ Yield Margins</h4>';
        html += '<div style="font-size:10px;color:var(--text-muted);margin-bottom:8px;">Peak stress vs material yield strength. Higher margin = safer.</div>';
        for (var sc in data.stress.yield_margins) {
            var sd = data.stress.component_stress[sc];
            var maxS = Math.round(Math.max.apply(null, sd.total_MPa));
            var margin = Math.round(data.stress.yield_margins[sc]);
            var yieldMPa = sd.yield_MPa;
            var status = margin > 30 ? 'success' : (margin > 10 ? 'warning' : 'danger');
            html += '<div class="info-row">';
            html += '<span class="info-label">' + capitalize(sc) + '</span>';
            html += '<span class="info-value ' + status + '">' + maxS + ' / ' + yieldMPa + ' MPa</span>';
            html += '</div>';
            html += '<div style="display:flex;align-items:center;margin:-2px 0 6px 0;gap:6px;">';
            html += '<div style="flex:1;height:4px;background:var(--bg-hover);border-radius:2px;">';
            var usePct = Math.min(Math.round(maxS / yieldMPa * 100), 100);
            html += '<div style="height:100%;width:' + usePct + '%;border-radius:2px;background:';
            html += status === 'success' ? 'var(--success)' : (status === 'warning' ? 'var(--warning)' : 'var(--danger)');
            html += ';"></div></div>';
            html += '<span style="font-size:9px;color:var(--text-muted);">' + margin + '% margin</span>';
            html += '</div>';
            html += '<div style="font-size:9px;color:var(--text-muted);margin:-2px 0 6px 12px;">Material: ' + sd.material + '</div>';
        }
        html += '</div>';

        html += '<div class="info-section"><h4>📊 Stress Legend</h4>';
        html += '<div style="font-size:10px;color:var(--text-muted);">';
        html += '• <span style="color:#1acc1a;">Green</span> = Low stress (safe)<br>';
        html += '• <span style="color:#cccc00;">Yellow</span> = Moderate stress<br>';
        html += '• <span style="color:#cc3300;">Red</span> = High stress (near yield)<br>';
        html += '• Sources: centrifugal + pressure + thermal';
        html += '</div></div>';
    }

    if (c.warnings && c.warnings.length > 0) {
        html += '<div class="info-section"><h4>⚠️ Warnings</h4>';
        c.warnings.forEach(function(w) { html += '<div class="info-row" style="color:var(--warning);font-size:11px;">\u26A0 ' + w + '</div>'; });
        html += '</div>';
    }

    el.innerHTML = html;
}

// ──── TPMS Lattice ────
async function toggleLattice() {
    state.latticeVisible = !state.latticeVisible;
    document.getElementById('btn-lattice').classList.toggle('active', state.latticeVisible);

    if (state.latticeVisible) {
        if (Object.keys(state.latticeMeshes).length === 0) {
            await loadLatticeMeshes();
        } else {
            for (var id in state.latticeMeshes) {
                state.latticeMeshes[id].visible = true;
            }
        }
        // Make solid components semi-transparent to reveal lattice
        for (var cid in state.components) {
            if (cid === 'combustor' || cid === 'nozzle') {
                state.components[cid].mesh.material.transparent = true;
                state.components[cid].mesh.material.opacity = 0.15;
                state.components[cid].mesh.material.needsUpdate = true;
            }
        }
        showBadge('TPMS LATTICE', '#FF6D00');
        renderLatticeInfo();
    } else {
        for (var id in state.latticeMeshes) {
            state.latticeMeshes[id].visible = false;
        }
        // Restore opacity
        for (var cid in state.components) {
            if (cid === 'combustor' || cid === 'nozzle') {
                var op = parseFloat(document.getElementById('slider-opacity').value);
                state.components[cid].mesh.material.transparent = op < 1;
                state.components[cid].mesh.material.opacity = op;
                state.components[cid].mesh.material.needsUpdate = true;
            }
        }
        document.getElementById('sim-badge').style.display = 'none';
        loadEngineData();
    }
}

async function loadLatticeMeshes(variation) {
    var varKey = variation || state.currentVariation || 'v1_gyroid_standard';
    state.currentVariation = varKey;

    document.getElementById('status-left').textContent = 'Loading TPMS lattice...';
    var loader = new STLLoader();

    // Determine file paths based on variation
    var basePath = varKey === 'default' ? '/stl/' : '/stl/variation/' + varKey + '/';
    var latticeFiles = [
        { id: 'combustor_lattice', file: basePath + 'combustor_lattice.stl', color: '#FF6D00', label: 'Combustor' },
        { id: 'nozzle_lattice',    file: basePath + 'nozzle_lattice.stl',    color: '#FF3D00', label: 'Nozzle' },
    ];

    for (var i = 0; i < latticeFiles.length; i++) {
        var lf = latticeFiles[i];
        try {
            var geometry = await new Promise(function(resolve, reject) {
                loader.load(lf.file, resolve, undefined, reject);
            });
            geometry.computeVertexNormals();

            var material = new THREE.MeshPhysicalMaterial({
                color: new THREE.Color(lf.color),
                metalness: 0.85,
                roughness: 0.2,
                clearcoat: 0.3,
                side: THREE.DoubleSide,
                clippingPlanes: state.sectionActive ? [state.clippingPlane] : [],
            });

            var mesh = new THREE.Mesh(geometry, material);
            mesh.castShadow = true;
            state.scene.add(mesh);
            state.latticeMeshes[lf.id] = mesh;

            var vCount = geometry.attributes.position.count;
            document.getElementById('status-left').textContent = 'Loaded ' + lf.label + ' (' + formatNum(Math.round(vCount)) + ' verts)';
        } catch(e) {
            console.warn('Lattice load failed: ' + lf.id, e);
        }
    }
    document.getElementById('status-left').textContent = 'TPMS lattice loaded';
}

async function switchVariation(varKey) {
    // Remove existing lattice meshes
    for (var id in state.latticeMeshes) {
        state.scene.remove(state.latticeMeshes[id]);
        if (state.latticeMeshes[id].geometry) state.latticeMeshes[id].geometry.dispose();
        if (state.latticeMeshes[id].material) state.latticeMeshes[id].material.dispose();
    }
    state.latticeMeshes = {};
    state.currentVariation = varKey;

    // Load new variation
    await loadLatticeMeshes(varKey);

    // Update active button
    document.querySelectorAll('.var-btn').forEach(function(btn) {
        btn.classList.toggle('active', btn.dataset.var === varKey);
    });

    // Re-render info panel
    renderLatticeInfo();
}

function renderLatticeInfo() {
    var el = document.getElementById('engine-info');
    var html = '';
    var varKey = state.currentVariation || 'v1_gyroid_standard';

    var variations = {
        'v1_gyroid_standard': { label: 'V1 Gyroid', desc: 'Balanced heat transfer & strength', type: 'Gyroid', eq: 'sin(x)cos(y)+sin(y)cos(z)+sin(z)cos(x)=t', cell: '14', color: '#FF6D00' },
        'v2_gyroid_dense':    { label: 'V2 Dense',  desc: 'Maximum heat exchange, heavier',    type: 'Gyroid (Dense)', eq: 'sin(x)cos(y)+sin(y)cos(z)+sin(z)cos(x)=t', cell: '10', color: '#FF8F00' },
        'v3_schwarz_p':       { label: 'V3 Schwarz', desc: 'Straight-through flow channels',   type: 'Schwarz-P',     eq: 'cos(x)+cos(y)+cos(z)=t',                  cell: '14', color: '#00E5FF' },
        'v4_diamond':         { label: 'V4 Diamond', desc: 'Highest stiffness-to-weight',      type: 'Diamond',       eq: 'sin·sin·sin + sin·cos·cos + ...',          cell: '14', color: '#76FF03' },
    };

    var cur = variations[varKey] || variations['v1_gyroid_standard'];

    // Variation selector buttons
    html += '<div class="info-section"><h4>🔬 TPMS Design Variants</h4>';
    html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;margin:6px 0;">';
    for (var vk in variations) {
        var v = variations[vk];
        var isActive = vk === varKey;
        var style = isActive
            ? 'background:' + v.color + '22;border:1px solid ' + v.color + ';color:' + v.color
            : 'background:var(--surface);border:1px solid var(--border);color:var(--text-secondary)';
        html += '<button class="var-btn' + (isActive ? ' active' : '') + '" data-var="' + vk + '" '
             + 'onclick="switchVariation(\'' + vk + '\')" '
             + 'style="' + style + ';padding:6px 4px;border-radius:6px;cursor:pointer;font-size:10px;font-weight:600;'
             + 'transition:all 0.2s;text-align:center;">'
             + v.label + '</button>';
    }
    html += '</div>';
    html += '<div style="color:var(--text-tertiary);font-size:10px;margin-top:4px;">' + cur.desc + '</div>';
    html += '</div>';

    // Current variant details
    html += '<div class="info-section"><h4>' + cur.type + '</h4>';
    html += infoRow('Equation', cur.eq);
    html += infoRow('Cell Size', cur.cell + ' mm');
    html += infoRow('Wall Type', 'Solid shell (thick walls)', 'accent');
    html += infoRow('Smoothing', 'Laplacian (2 iterations)');
    html += '</div>';

    html += '<div class="info-section"><h4>Benefits</h4>';
    html += '<div class="info-row" style="color:var(--success);">✓ Up to 60% weight reduction</div>';
    html += '<div class="info-row" style="color:var(--success);">✓ 3× surface area for heat transfer</div>';
    html += '<div class="info-row" style="color:var(--success);">✓ Self-supporting for metal 3D printing</div>';
    html += '<div class="info-row" style="color:var(--success);">✓ Contained within engine walls</div>';
    html += '<div class="info-row" style="color:var(--success);">✓ No overhangs > 45° (DMLS compatible)</div>';
    html += '</div>';

    html += '<div class="info-section"><h4>Manufacturing</h4>';
    html += infoRow('Process', 'DMLS / SLM Metal 3D Printing');
    html += infoRow('Material', 'Inconel 718 (combustor)');
    html += infoRow('Min Feature', '0.4 mm wall thickness');
    html += infoRow('Print Orient.', 'Vertical (Z-axis = engine axis)');
    html += '</div>';

    el.innerHTML = html;
}

// Expose to global scope for inline onclick handlers
window.switchVariation = switchVariation;

// Start
init();
