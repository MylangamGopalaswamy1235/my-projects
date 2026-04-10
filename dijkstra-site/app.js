/* ================================================================
   Smart Paste Collection Root Optimizer — app.js
   Pure JavaScript implementation mirroring the C source (dijkstra.c)
   ================================================================ */

"use strict";

const INF = Infinity;

/* ── Dijkstra (mirrors dijkstra.c logic exactly) ── */
function dijkstra(adj, V, src) {
    const dist   = new Array(V).fill(INF);
    const parent = new Array(V).fill(-1);
    const visited = new Array(V).fill(false);

    dist[src] = 0;

    /* Min-heap via a simple priority-queue array (small V, acceptable) */
    const pq = [[0, src]];

    while (pq.length > 0) {
        /* Extract min */
        let minIdx = 0;
        for (let i = 1; i < pq.length; i++)
            if (pq[i][0] < pq[minIdx][0]) minIdx = i;
        const [d, u] = pq.splice(minIdx, 1)[0];

        if (visited[u]) continue;
        visited[u] = true;

        for (const [v, w] of (adj[u] || [])) {
            if (!visited[v] && dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                parent[v] = u;
                pq.push([dist[v], v]);
            }
        }
    }

    let totalCost = 0, reachable = 0;
    for (let i = 0; i < V; i++) {
        if (i !== src && dist[i] < INF) { totalCost += dist[i]; reachable++; }
    }
    return { dist, parent, totalCost, reachable };
}

function buildPath(parent, j) {
    const path = [];
    let cur = j;
    while (cur !== -1) { path.unshift(cur); cur = parent[cur]; }
    return path;
}

function buildAdj(V, edges) {
    const adj = Array.from({length: V}, () => []);
    for (const {src, dst, w} of edges) {
        if (src < 0 || src >= V || dst < 0 || dst >= V) continue;
        adj[src].push([dst, w]);
        adj[dst].push([src, w]);
    }
    return adj;
}

/* ── Demo data ── */
const DEMO = {
    nodes: 6,
    edges: [
        {src:0, dst:1, w:4}, {src:0, dst:2, w:2}, {src:1, dst:2, w:5},
        {src:1, dst:3, w:10},{src:2, dst:4, w:3}, {src:4, dst:3, w:4},
        {src:3, dst:5, w:11}
    ]
};

/* ── Canvas drawing ── */
const canvas = document.getElementById('graphCanvas');
const ctx = canvas.getContext('2d');

function nodePositions(V, W, H) {
    const positions = [];
    const cx = W / 2, cy = H / 2;
    const r  = Math.min(W, H) * 0.38;
    for (let i = 0; i < V; i++) {
        const angle = (2 * Math.PI * i / V) - Math.PI / 2;
        positions.push([cx + r * Math.cos(angle), cy + r * Math.sin(angle)]);
    }
    return positions;
}

function drawGraph(V, edges, root, distResult) {
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    if (V === 0) return;
    const pos = nodePositions(V, W, H);
    const R   = Math.max(16, Math.min(22, 200 / V));
    const style = getComputedStyle(document.documentElement);
    const borderCol = '#2a3148';

    /* Draw edges */
    const edgesDrawn = new Set();
    for (const {src, dst, w} of edges) {
        if (src < 0 || src >= V || dst < 0 || dst >= V) continue;
        const key = [Math.min(src,dst), Math.max(src,dst)].join('-');
        if (edgesDrawn.has(key)) continue;
        edgesDrawn.add(key);

        let isOptimalEdge = false;
        if (distResult) {
            const {parent} = distResult;
            if (parent[dst] === src || parent[src] === dst) isOptimalEdge = true;
        }

        ctx.beginPath();
        ctx.moveTo(pos[src][0], pos[src][1]);
        ctx.lineTo(pos[dst][0], pos[dst][1]);
        ctx.strokeStyle = isOptimalEdge ? '#3b82f6' : borderCol;
        ctx.lineWidth   = isOptimalEdge ? 2.5 : 1.5;
        ctx.stroke();

        /* Weight label */
        const mx = (pos[src][0] + pos[dst][0]) / 2;
        const my = (pos[src][1] + pos[dst][1]) / 2;
        ctx.fillStyle = '#7f8ea3';
        ctx.font = `bold ${Math.max(10, R - 4)}px monospace`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = isOptimalEdge ? '#60a5fa' : '#7f8ea3';
        ctx.fillText(w, mx, my - 8);
    }

    /* Draw nodes */
    for (let i = 0; i < V; i++) {
        const [x, y] = pos[i];
        const isRoot    = i === root;
        const reachable = distResult && distResult.dist[i] < INF;
        const dist      = distResult ? distResult.dist[i] : null;

        /* Outer glow for root */
        if (isRoot) {
            ctx.beginPath();
            ctx.arc(x, y, R + 6, 0, 2 * Math.PI);
            ctx.fillStyle = 'rgba(34,211,238,.18)';
            ctx.fill();
        }

        /* Node circle */
        ctx.beginPath();
        ctx.arc(x, y, R, 0, 2 * Math.PI);
        if (isRoot)           ctx.fillStyle = '#22d3ee';
        else if (reachable)   ctx.fillStyle = '#3b82f6';
        else                  ctx.fillStyle = '#1e2535';
        ctx.fill();
        ctx.strokeStyle = isRoot ? '#67e8f9' : reachable ? '#60a5fa' : borderCol;
        ctx.lineWidth   = 2;
        ctx.stroke();

        /* Node label */
        ctx.fillStyle = '#fff';
        ctx.font = `bold ${Math.max(10, R - 2)}px sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(i, x, y);

        /* Distance badge */
        if (distResult && !isRoot && dist < INF) {
            ctx.fillStyle = 'rgba(245,158,11,.9)';
            ctx.font = `${Math.max(9, R - 5)}px monospace`;
            ctx.fillText(`d=${dist}`, x, y + R + 10);
        } else if (distResult && isRoot) {
            ctx.fillStyle = 'rgba(34,211,238,.85)';
            ctx.font = `${Math.max(9, R - 5)}px monospace`;
            ctx.fillText('root', x, y + R + 10);
        }
    }
}

/* ── Edge input generation ── */
let currentEdgeCount = 0;

function generateEdgeInputs(count) {
    const container = document.getElementById('edgeInputs');
    container.innerHTML = '';
    if (count <= 0) return;

    const header = document.createElement('div');
    header.className = 'edge-row-header';
    header.innerHTML = '<span>Source</span><span>Dest</span><span>Weight</span>';
    container.appendChild(header);

    for (let i = 0; i < count; i++) {
        const row = document.createElement('div');
        row.className = 'edge-row';
        row.innerHTML = `
            <div><input type="number" class="edge-src" min="0" placeholder="0" /></div>
            <div><input type="number" class="edge-dst" min="0" placeholder="1" /></div>
            <div><input type="number" class="edge-w" min="0" placeholder="1" /></div>`;
        container.appendChild(row);
    }
    currentEdgeCount = count;
}

function readEdges() {
    const rows = document.querySelectorAll('.edge-row');
    const edges = [];
    rows.forEach(row => {
        const src = parseInt(row.querySelector('.edge-src').value);
        const dst = parseInt(row.querySelector('.edge-dst').value);
        const w   = parseInt(row.querySelector('.edge-w').value);
        if (!isNaN(src) && !isNaN(dst) && !isNaN(w) && w >= 0) {
            edges.push({src, dst, w});
        }
    });
    return edges;
}

function fillEdgeInputs(edgeData) {
    const rows = document.querySelectorAll('.edge-row');
    edgeData.forEach((e, i) => {
        if (!rows[i]) return;
        rows[i].querySelector('.edge-src').value = e.src;
        rows[i].querySelector('.edge-dst').value = e.dst;
        rows[i].querySelector('.edge-w').value   = e.w;
    });
}

/* ── Result rendering ── */
function renderResult(V, edges, root, distResult, allResults) {
    document.getElementById('placeholder').style.display = 'none';
    const section = document.getElementById('resultSection');
    section.style.display = 'block';

    /* Badge */
    const badge = document.getElementById('resultBadge');
    badge.textContent = `✓ Optimal Root: Node ${root} — Total Cost: ${distResult.totalCost}`;

    /* Stats */
    const statsGrid = document.getElementById('statsGrid');
    statsGrid.innerHTML = `
        <div class="stat-card">
            <div class="stat-label">Optimal Root</div>
            <div class="stat-value">${root}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Total Cost</div>
            <div class="stat-value yellow">${distResult.totalCost}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Reachable</div>
            <div class="stat-value green">${distResult.reachable} / ${V - 1}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Nodes</div>
            <div class="stat-value">${V}</div>
        </div>`;

    /* Path table */
    const tbody = document.getElementById('pathTableBody');
    tbody.innerHTML = '';
    for (let i = 0; i < V; i++) {
        if (i === root) continue;
        const d = distResult.dist[i];
        const reachable = d < INF;
        const pathStr = reachable ? buildPath(distResult.parent, i).join(' → ') : '—';
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>Node ${i}</td>
            <td>${reachable ? d : '∞'}</td>
            <td>${pathStr}</td>
            <td><span class="badge ${reachable ? 'badge-green' : 'badge-red'}">${reachable ? 'Reachable' : 'Unreachable'}</span></td>`;
        tbody.appendChild(tr);
    }

    /* Comparison table */
    if (allResults) {
        const sorted = [...allResults].sort((a, b) => {
            if (b.reachable !== a.reachable) return b.reachable - a.reachable;
            return a.totalCost - b.totalCost;
        });
        const rankOf = {};
        sorted.forEach((r, idx) => { rankOf[r.root] = idx + 1; });

        const cbody = document.getElementById('compTableBody');
        cbody.innerHTML = '';
        sorted.forEach(r => {
            const rank = rankOf[r.root];
            let rankBadge;
            if (rank === 1) rankBadge = '<span class="badge badge-gold">🥇 1st</span>';
            else if (rank === 2) rankBadge = '<span class="badge badge-blue">2nd</span>';
            else rankBadge = `<span class="badge">#${rank}</span>`;

            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${r.root === root ? `<strong>Node ${r.root} ★</strong>` : `Node ${r.root}`}</td>
                <td>${r.totalCost < INF ? r.totalCost : '—'}</td>
                <td>${r.reachable} / ${V - 1}</td>
                <td>${rankBadge}</td>`;
            cbody.appendChild(tr);
        });
    }

    /* Redraw graph with result highlights */
    drawGraph(V, edges, root, distResult);
}

/* ── Main run ── */
function run() {
    const V         = parseInt(document.getElementById('numNodes').value);
    const mode      = document.querySelector('input[name="mode"]:checked').value;
    const edges     = readEdges();

    if (isNaN(V) || V < 2 || V > 20) {
        alert('Please enter a valid node count (2–20).'); return;
    }
    if (edges.length === 0) {
        alert('Please add at least one valid edge.'); return;
    }
    for (const {src, dst} of edges) {
        if (src >= V || dst >= V) {
            alert(`Edge references node ${Math.max(src,dst)} but max node is ${V-1}.`); return;
        }
    }

    const adj = buildAdj(V, edges);

    /* Run Dijkstra for all roots (for comparison table) */
    const allResults = [];
    for (let r = 0; r < V; r++) {
        const res = dijkstra(adj, V, r);
        allResults.push({root: r, ...res});
    }

    let chosenRoot;
    if (mode === 'auto') {
        /* Pick root with max reachability, then min total cost */
        let best = allResults[0];
        for (const r of allResults) {
            if (r.reachable > best.reachable ||
               (r.reachable === best.reachable && r.totalCost < best.totalCost))
                best = r;
        }
        chosenRoot = best.root;
    } else {
        chosenRoot = parseInt(document.getElementById('manualRoot').value);
        if (isNaN(chosenRoot) || chosenRoot < 0 || chosenRoot >= V) {
            alert(`Root must be between 0 and ${V - 1}.`); return;
        }
    }

    const distResult = allResults.find(r => r.root === chosenRoot);
    renderResult(V, edges, chosenRoot, distResult, allResults);
}

/* ── C source code display ── */
const C_SOURCE = `/*
 * Smart Paste Collection Root Optimizer
 * Using Dijkstra's Algorithm
 *
 * Compile: gcc -o dijkstra dijkstra.c -lm
 * Usage:   ./dijkstra
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>

#define MAX_NODES 100
#define INF INT_MAX

typedef struct Edge {
    int dest;
    int weight;
    struct Edge* next;
} Edge;

typedef struct Graph {
    int numNodes;
    Edge* adjList[MAX_NODES];
} Graph;

/* ... (Min-heap, Dijkstra, findOptimalRoot, main) ... */
/* See dijkstra.c for the complete source code        */

/* findOptimalRoot() runs Dijkstra from every node,
   sums all shortest-path distances, and returns
   the root with the minimum total collection cost.  */

int findOptimalRoot(Graph* graph) {
    int bestRoot = 0, bestCost = INT_MAX;
    for (int r = 0; r < graph->numNodes; r++) {
        Result res = dijkstra(graph, r);
        if (res.reachable == graph->numNodes - 1
            && res.totalCost < bestCost) {
            bestCost = res.totalCost;
            bestRoot = r;
        }
    }
    return bestRoot;
}`;

/* ── Event bindings ── */
document.getElementById('btnGenEdges').addEventListener('click', () => {
    const n = parseInt(document.getElementById('numEdges').value);
    if (isNaN(n) || n < 1 || n > 100) { alert('Edges: 1–100'); return; }
    generateEdgeInputs(n);
});

document.getElementById('btnRun').addEventListener('click', run);

document.getElementById('btnLoadDemo').addEventListener('click', () => {
    document.getElementById('numNodes').value = DEMO.nodes;
    document.getElementById('numEdges').value = DEMO.edges.length;
    generateEdgeInputs(DEMO.edges.length);
    fillEdgeInputs(DEMO.edges);
    document.querySelector('input[name="mode"][value="auto"]').checked = true;
    document.getElementById('manualRootGroup').style.display = 'none';
    drawGraph(DEMO.nodes, DEMO.edges, null, null);
});

document.querySelectorAll('input[name="mode"]').forEach(r => {
    r.addEventListener('change', () => {
        const manual = document.querySelector('input[name="mode"]:checked').value === 'manual';
        document.getElementById('manualRootGroup').style.display = manual ? 'block' : 'none';
    });
});

document.getElementById('btnToggleCode').addEventListener('click', function() {
    const block = document.getElementById('codeBlock');
    const pre   = document.getElementById('cSourcePre');
    const show  = block.style.display === 'none';
    block.style.display = show ? 'block' : 'none';
    this.textContent = (show ? '▲ Hide' : '▼ Show') + ' Code';
    if (show && !pre.textContent.trim()) {
        /* Load actual dijkstra.c if same-origin, else show excerpt */
        fetch('dijkstra.c')
            .then(r => r.ok ? r.text() : null)
            .then(t => { pre.textContent = t || C_SOURCE; })
            .catch(() => { pre.textContent = C_SOURCE; });
    }
});

/* ── Init ── */
(function init() {
    generateEdgeInputs(DEMO.edges.length);
    document.getElementById('numEdges').value = DEMO.edges.length;
    fillEdgeInputs(DEMO.edges);
    document.getElementById('numNodes').value = DEMO.nodes;
    drawGraph(DEMO.nodes, DEMO.edges, null, null);
})();
