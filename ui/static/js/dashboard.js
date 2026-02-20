/**
 * NovaTurbo Engineering Dashboard
 * Interactive Brayton cycle charts, Pareto front, and design parameter controls.
 * Uses Chart.js for plotting.
 */

import { Chart, registerables } from 'chart.js';
Chart.register(...registerables);

// Chart color theme
const C = {
    accent: '#00E5FF',
    accentDim: 'rgba(0,229,255,0.15)',
    danger: '#ef4444',
    warning: '#f59e0b',
    success: '#22c55e',
    blue: '#3b82f6',
    purple: '#a78bfa',
    grid: 'rgba(255,255,255,0.06)',
    gridLabel: '#64748b',
    bg: '#151c2c',
    surface: '#1e293b',
    text: '#e2e8f0',
};

let tsChart = null;
let stationChart = null;
let paretoChart = null;
let currentParams = {
    pr: 3.5, tit: 1100, eta_c: 0.78, eta_t: 0.82, mdot: 0.15, mach: 0,
    nozzle_exit_mm: 55, nozzle_length_mm: 50,
    combustor_outer_mm: 110, combustor_inner_mm: 60, combustor_length_mm: 80,
};
let currentTargets = {
    target_thrust_N: 100,
    target_tsfc_g_kNs: 35,
};
let debounceTimer = null;
let learningPollHandle = null;
const cycleParamKeys = ['pr', 'tit', 'eta_c', 'eta_t', 'mdot', 'mach'];

// ─── Public API ───
export function openDashboard() {
    const overlay = document.getElementById('dashboard-overlay');
    overlay.classList.add('visible');
    const btn = document.getElementById('btn-dashboard');
    if (btn) btn.classList.add('active');
    // Load data on first open
    if (!tsChart) {
        initCharts();
        fetchBrayton();
        fetchPareto();
    }
    refreshLearningStatus();
}

export function closeDashboard() {
    document.getElementById('dashboard-overlay').classList.remove('visible');
    const btn = document.getElementById('btn-dashboard');
    if (btn) btn.classList.remove('active');
}

export function toggleDashboard() {
    const overlay = document.getElementById('dashboard-overlay');
    if (overlay.classList.contains('visible')) closeDashboard();
    else openDashboard();
}

// ─── Chart Setup ───
function chartDefaults() {
    return {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 400 },
        plugins: {
            legend: { labels: { color: C.gridLabel, font: { size: 10, family: 'Inter' } } },
            tooltip: {
                backgroundColor: C.bg,
                borderColor: C.accent,
                borderWidth: 1,
                titleFont: { family: 'Inter', size: 11 },
                bodyFont: { family: 'JetBrains Mono', size: 10 },
                titleColor: C.text,
                bodyColor: C.gridLabel,
            },
        },
        scales: {
            x: { ticks: { color: C.gridLabel, font: { size: 10 } }, grid: { color: C.grid } },
            y: { ticks: { color: C.gridLabel, font: { size: 10 } }, grid: { color: C.grid } },
        },
    };
}

function initCharts() {
    // T-s Diagram
    const tsCtx = document.getElementById('chart-ts').getContext('2d');
    tsChart = new Chart(tsCtx, {
        type: 'line',
        data: {
            datasets: [{
                label: 'Brayton Cycle',
                data: [],
                borderColor: C.accent,
                backgroundColor: C.accentDim,
                fill: true,
                tension: 0.3,
                pointRadius: 5,
                pointBackgroundColor: [C.blue, C.blue, C.success, C.danger, C.warning, C.purple],
                pointBorderColor: '#fff',
                pointBorderWidth: 1,
                borderWidth: 2,
            }],
        },
        options: {
            ...chartDefaults(),
            plugins: {
                ...chartDefaults().plugins,
                legend: { display: false },
                tooltip: {
                    ...chartDefaults().plugins.tooltip,
                    callbacks: {
                        label: ctx => {
                            const p = ctx.raw;
                            return `${p.label}: T=${p.y} K, s=${p.x} J/kg·K`;
                        }
                    }
                },
            },
            scales: {
                x: {
                    type: 'linear',
                    title: { display: true, text: 'Entropy (J/kg·K)', color: C.gridLabel, font: { size: 10 } },
                    ticks: { color: C.gridLabel, font: { size: 9 } },
                    grid: { color: C.grid },
                },
                y: {
                    title: { display: true, text: 'Temperature (K)', color: C.gridLabel, font: { size: 10 } },
                    ticks: { color: C.gridLabel, font: { size: 9 } },
                    grid: { color: C.grid },
                },
            },
        },
    });

    // Station bar chart (dual-axis: Temp + Pressure)
    const stCtx = document.getElementById('chart-stations').getContext('2d');
    stationChart = new Chart(stCtx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Temperature (K)',
                    data: [],
                    backgroundColor: 'rgba(239,68,68,0.6)',
                    borderColor: C.danger,
                    borderWidth: 1,
                    yAxisID: 'y',
                    borderRadius: 3,
                },
                {
                    label: 'Pressure (kPa)',
                    data: [],
                    backgroundColor: 'rgba(59,130,246,0.6)',
                    borderColor: C.blue,
                    borderWidth: 1,
                    yAxisID: 'y1',
                    borderRadius: 3,
                },
            ],
        },
        options: {
            ...chartDefaults(),
            scales: {
                x: { ticks: { color: C.gridLabel, font: { size: 9 } }, grid: { display: false } },
                y: {
                    position: 'left',
                    title: { display: true, text: 'Temp (K)', color: '#ef4444', font: { size: 9 } },
                    ticks: { color: '#ef4444', font: { size: 9 } },
                    grid: { color: C.grid },
                },
                y1: {
                    position: 'right',
                    title: { display: true, text: 'Pressure (kPa)', color: '#3b82f6', font: { size: 9 } },
                    ticks: { color: '#3b82f6', font: { size: 9 } },
                    grid: { drawOnChartArea: false },
                },
            },
        },
    });

    // Pareto scatter
    const pCtx = document.getElementById('chart-pareto').getContext('2d');
    paretoChart = new Chart(pCtx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Designs (T/W color)',
                data: [],
                backgroundColor: [],
                pointRadius: 3.5,
                pointHoverRadius: 6,
            }],
        },
        options: {
            ...chartDefaults(),
            plugins: {
                ...chartDefaults().plugins,
                legend: { display: false },
                tooltip: {
                    ...chartDefaults().plugins.tooltip,
                    callbacks: {
                        label: ctx => {
                            const p = ctx.raw;
                            return `Thrust: ${p.x}N | Mass: ${p.y}kg | T/W: ${p.tw}`;
                        }
                    }
                },
            },
            scales: {
                x: {
                    title: { display: true, text: 'Thrust (N)', color: C.gridLabel, font: { size: 10 } },
                    ticks: { color: C.gridLabel, font: { size: 9 } },
                    grid: { color: C.grid },
                },
                y: {
                    title: { display: true, text: 'Mass (kg)', color: C.gridLabel, font: { size: 10 } },
                    ticks: { color: C.gridLabel, font: { size: 9 } },
                    grid: { color: C.grid },
                },
            },
        },
    });

    // Bind sliders
    document.querySelectorAll('.dash-slider').forEach(slider => {
        slider.addEventListener('input', onSliderChange);
    });

    // Close button
    document.getElementById('dash-close').addEventListener('click', closeDashboard);
    document.getElementById('btn-apply-model').addEventListener('click', applyModelChanges);
    document.getElementById('btn-save-variant').addEventListener('click', saveVariantSample);
    document.getElementById('btn-run-cfd').addEventListener('click', runCfdCalibration);
    document.getElementById('btn-train-ui').addEventListener('click', startUiTraining);
    document.getElementById('btn-inverse-design').addEventListener('click', runInverseDesign);
    document.getElementById('target-thrust').addEventListener('input', onTargetChange);
    document.getElementById('target-tsfc').addEventListener('input', onTargetChange);
    onTargetChange();
    // ESC key
    document.addEventListener('keydown', e => {
        if (e.key === 'Escape') closeDashboard();
    });

    if (!learningPollHandle) {
        learningPollHandle = setInterval(refreshLearningStatus, 3000);
    }
}

// ─── Data Fetching ───
async function fetchBrayton() {
    const cycleParams = {};
    cycleParamKeys.forEach(k => { cycleParams[k] = currentParams[k]; });
    const qs = new URLSearchParams(cycleParams).toString();
    try {
        const resp = await fetch('/api/brayton/compute?' + qs);
        const data = await resp.json();
        updateTsChart(data.ts_diagram);
        updateStationChart(data.labels, data.temperatures, data.pressures);
        updateKpis(data.performance, data.warnings, data.valid);
    } catch (e) {
        console.warn('Brayton fetch failed:', e);
    }
}

async function fetchPareto() {
    try {
        const resp = await fetch('/api/pareto');
        const data = await resp.json();
        updateParetoChart(data.points, data.stats);
    } catch (e) {
        console.warn('Pareto fetch failed:', e);
    }
}

// ─── Chart Updates ───
function updateTsChart(tsPoints) {
    if (!tsChart || !tsPoints) return;
    // Close the cycle loop: add first point at end
    const pts = tsPoints.map(p => ({ x: p.s, y: p.T, label: p.label }));
    if (pts.length > 1) pts.push({ ...pts[0] });

    tsChart.data.datasets[0].data = pts;
    tsChart.data.datasets[0].pointBackgroundColor =
        pts.map((_, i) => [C.blue, C.blue, C.success, C.danger, C.warning, C.purple, C.blue][i] || C.accent);
    tsChart.update();
}

function updateStationChart(labels, temps, pressures) {
    if (!stationChart) return;
    stationChart.data.labels = labels;
    stationChart.data.datasets[0].data = temps;
    stationChart.data.datasets[1].data = pressures;
    stationChart.update();
}

function updateKpis(perf, warnings, valid) {
    if (!perf) return;
    const set = (id, val, unit) => {
        const el = document.getElementById(id);
        if (el) el.innerHTML = `<span class="kpi-val">${val}</span><span class="kpi-unit">${unit}</span>`;
    };
    set('kpi-thrust', perf.thrust_N, 'N');
    set('kpi-tsfc', perf.tsfc_g_kNs, 'g/kN·s');
    set('kpi-eff', perf.thermal_eff, '%');
    set('kpi-exhaust', perf.exhaust_vel, 'm/s');
    set('kpi-fuel', perf.fuel_flow_g_hr, 'g/hr');
    set('kpi-afr', perf.air_fuel_ratio, ':1');

    // Status indicator
    const statusEl = document.getElementById('dash-status');
    if (statusEl) {
        if (!valid) {
            statusEl.textContent = '⚠ ' + (warnings[0] || 'Invalid design');
            statusEl.className = 'dash-status warn';
        } else if (warnings.length > 0) {
            statusEl.textContent = '⚡ ' + warnings[0];
            statusEl.className = 'dash-status caution';
        } else {
            statusEl.textContent = '✓ Design within limits';
            statusEl.className = 'dash-status ok';
        }
    }
}

function setDashStatus(text, cls) {
    const statusEl = document.getElementById('dash-status');
    if (!statusEl) return;
    statusEl.textContent = text;
    statusEl.className = 'dash-status ' + cls;
}

function twColor(tw, minTw, maxTw) {
    const t = Math.max(0, Math.min(1, (tw - minTw) / (maxTw - minTw + 0.01)));
    // Blue (low T/W) → Cyan → Green → Yellow → Red (high T/W)
    if (t < 0.25) return `rgba(59,130,246,0.7)`;
    if (t < 0.5)  return `rgba(0,229,255,0.7)`;
    if (t < 0.75) return `rgba(34,197,94,0.7)`;
    return `rgba(245,158,11,0.7)`;
}

function updateParetoChart(points, stats) {
    if (!paretoChart || !points) return;
    const minTw = stats.tw_range[0];
    const maxTw = stats.tw_range[1];

    paretoChart.data.datasets[0].data = points.map(p => ({
        x: p.thrust, y: p.mass, tw: p.tw,
    }));
    paretoChart.data.datasets[0].backgroundColor = points.map(p => twColor(p.tw, minTw, maxTw));
    paretoChart.update();

    // Update Pareto legend counts
    const infoEl = document.getElementById('pareto-info');
    if (infoEl) {
        infoEl.textContent = `${points.length} sampled designs | T/W range ${stats.tw_range[0]} - ${stats.tw_range[1]}`;
    }
}

async function saveVariantSample() {
    const btn = document.getElementById('btn-save-variant');
    if (!btn) return;
    const old = btn.textContent;
    btn.disabled = true;
    btn.textContent = 'Saving...';
    try {
        const resp = await fetch('/api/learning/add_variant', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(currentParams),
        });
        const data = await resp.json();
        if (!resp.ok || data.error) throw new Error(data.error || 'Save failed');
        setDashStatus('✓ Variant sample saved', 'ok');
        const info = document.getElementById('learning-status');
        if (info) {
            info.textContent = `Saved samples: ${data.variant_count} | latest thrust ${data.sample.thrust_N} N`;
        }
    } catch (err) {
        setDashStatus('⚠ ' + err.message, 'warn');
    } finally {
        btn.disabled = false;
        btn.textContent = old;
    }
}

async function startUiTraining() {
    const btn = document.getElementById('btn-train-ui');
    if (!btn) return;
    const old = btn.textContent;
    btn.disabled = true;
    btn.textContent = 'Starting...';
    try {
        const useCfd = document.getElementById('chk-use-cfd')?.checked ?? false;
        const resp = await fetch('/api/learning/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ epochs: 80, use_cfd: useCfd }),
        });
        const data = await resp.json();
        if (!resp.ok || data.error) throw new Error(data.error || 'Training start failed');
        setDashStatus('⏳ UI training started', 'caution');
        const info = document.getElementById('learning-status');
        if (info) {
            const mode = useCfd ? 'CFD-calibrated' : 'physics-only';
            info.textContent = `Training (${mode})... samples: ${data.variant_count}, epochs: ${data.epochs}`;
        }
    } catch (err) {
        setDashStatus('⚠ ' + err.message, 'warn');
    } finally {
        btn.disabled = false;
        btn.textContent = old;
    }
}

async function refreshLearningStatus() {
    try {
        const resp = await fetch('/api/learning/status');
        const data = await resp.json();
        const info = document.getElementById('learning-status');
        if (info) {
            const cfd = data.cfd || {};
            let trainText = `Saved UI samples: ${data.variant_count}`;
            if (data.state === 'running') {
                trainText = `Training running... samples: ${data.variant_count}`;
            } else if (data.state === 'done') {
                trainText = `Training complete. Samples: ${data.variant_count}`;
            } else if (data.state === 'failed') {
                trainText = `Training failed: ${data.message}`;
            }

            let cfdText = '';
            if (cfd.state === 'running') {
                cfdText = ` | CFD running (${cfd.solver || 'solver'}) ${cfd.processed || 0} cases`;
            } else if (cfd.state === 'done') {
                cfdText = ` | CFD ready (${cfd.sample_count || 0} samples)`;
            } else if (cfd.state === 'failed') {
                cfdText = ` | CFD failed: ${cfd.message}`;
            } else if (data.has_cfd_calibration) {
                cfdText = ' | CFD calibration available';
            }
            info.textContent = trainText + cfdText;
        }
    } catch (_e) {
        // Keep UI silent if status polling fails intermittently.
    }
}

function onTargetChange() {
    const thrustEl = document.getElementById('target-thrust');
    const tsfcEl = document.getElementById('target-tsfc');
    currentTargets.target_thrust_N = parseFloat(thrustEl.value);
    currentTargets.target_tsfc_g_kNs = parseFloat(tsfcEl.value);
    document.getElementById('target-thrust-val').textContent = String(Math.round(currentTargets.target_thrust_N));
    document.getElementById('target-tsfc-val').textContent = String(Math.round(currentTargets.target_tsfc_g_kNs));
}

function setUiParamValue(paramKey, value) {
    if (value === undefined || value === null) return;
    currentParams[paramKey] = value;
    const slider = document.querySelector(`.dash-slider[data-param="${paramKey}"]`);
    if (slider) slider.value = String(value);
    const label = document.getElementById('val-' + paramKey);
    if (label) {
        const v = parseFloat(value);
        label.textContent = Number.isInteger(v) ? String(v) : v.toFixed(2).replace(/\.00$/, '');
    }
}

async function runCfdCalibration() {
    const btn = document.getElementById('btn-run-cfd');
    if (!btn) return;
    const solver = document.getElementById('cfd-solver')?.value || 'openfoam';
    const old = btn.textContent;
    btn.disabled = true;
    btn.textContent = 'Starting CFD...';
    try {
        const resp = await fetch('/api/learning/cfd/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ solver, max_cases: 5, timeout_s: 900 }),
        });
        const data = await resp.json();
        if (!resp.ok || data.error) throw new Error(data.error || 'CFD calibration failed to start');
        setDashStatus(`⏳ ${solver.toUpperCase()} calibration started`, 'caution');
        const info = document.getElementById('learning-status');
        if (info) info.textContent = `CFD calibration running with ${solver}...`;
    } catch (err) {
        setDashStatus('⚠ ' + err.message, 'warn');
    } finally {
        btn.disabled = false;
        btn.textContent = old;
    }
}

async function runInverseDesign() {
    const btn = document.getElementById('btn-inverse-design');
    if (!btn) return;
    onTargetChange();

    const old = btn.textContent;
    btn.disabled = true;
    btn.textContent = 'Searching...';
    try {
        const resp = await fetch('/api/inverse_design', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                target_thrust_N: currentTargets.target_thrust_N,
                target_tsfc_g_kNs: currentTargets.target_tsfc_g_kNs,
                n_samples: 3000,
                top_k: 5,
            }),
        });
        const data = await resp.json();
        if (!resp.ok || data.error) throw new Error(data.error || 'Inverse design failed');

        if (data.best_ui_params) {
            Object.entries(data.best_ui_params).forEach(([k, v]) => setUiParamValue(k, v));
            fetchBrayton();
        }

        const inv = document.getElementById('inverse-status');
        if (inv && data.best && data.best.performance) {
            const p = data.best.performance;
            inv.textContent = `Best match: thrust ${p.thrust_N.toFixed(1)} N, TSFC ${p.tsfc_g_kNs.toFixed(1)} g/kN·s`;
        }
        setDashStatus('✓ Inverse design suggestion ready', 'ok');
    } catch (err) {
        setDashStatus('⚠ ' + err.message, 'warn');
    } finally {
        btn.disabled = false;
        btn.textContent = old;
    }
}

// ─── Slider Handling ───
function onSliderChange(e) {
    const slider = e.target;
    const key = slider.dataset.param;
    const val = parseFloat(slider.value);
    currentParams[key] = val;

    // Update label
    const label = document.getElementById('val-' + key);
    if (label) label.textContent = Number.isInteger(val) ? val : val.toFixed(2).replace(/\.00$/, '');

    // Update Brayton charts only for cycle parameters
    if (cycleParamKeys.includes(key)) {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => fetchBrayton(), 200);
    }
}

async function applyModelChanges() {
    const btn = document.getElementById('btn-apply-model');
    if (!btn) return;

    const prevText = btn.textContent;
    btn.disabled = true;
    btn.textContent = 'Applying...';

    try {
        const resp = await fetch('/api/model/regenerate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(currentParams),
        });
        const data = await resp.json();
        if (!resp.ok || data.error) {
            throw new Error(data.error || 'Model regeneration failed');
        }

        const statusEl = document.getElementById('dash-status');
        if (statusEl) {
            statusEl.textContent = '✓ 3D model regenerated';
            statusEl.className = 'dash-status ok';
        }

        // Ensure KPIs reflect the regenerated configuration before reload.
        if (data.performance) {
            updateKpis(
                {
                    thrust_N: data.performance.thrust_N,
                    tsfc_g_kNs: data.performance.tsfc_g_kNs,
                    thermal_eff: data.performance.thermal_eff,
                    exhaust_vel: '-',
                    fuel_flow_g_hr: '-',
                    air_fuel_ratio: '-',
                },
                data.warnings || [],
                true
            );
        }

        setTimeout(() => window.location.reload(), 300);
    } catch (err) {
        const statusEl = document.getElementById('dash-status');
        if (statusEl) {
            statusEl.textContent = '⚠ ' + err.message;
            statusEl.className = 'dash-status warn';
        }
        btn.disabled = false;
        btn.textContent = prevText;
    }
}
