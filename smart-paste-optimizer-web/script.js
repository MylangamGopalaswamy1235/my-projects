const cCode = `#include <stdio.h>
#include <limits.h>

#define MAX_NODES 100
#define INF 1000000000

void dijkstra(int n, int graph[MAX_NODES][MAX_NODES], int source, int dist[MAX_NODES]) {
    int visited[MAX_NODES] = {0};

    for (int i = 0; i < n; i++) {
        dist[i] = INF;
    }
    dist[source] = 0;

    for (int count = 0; count < n - 1; count++) {
        int u = -1;
        int best = INF;

        for (int i = 0; i < n; i++) {
            if (!visited[i] && dist[i] < best) {
                best = dist[i];
                u = i;
            }
        }

        if (u == -1) break;
        visited[u] = 1;

        for (int v = 0; v < n; v++) {
            if (!visited[v] && graph[u][v] < INF && dist[u] + graph[u][v] < dist[v]) {
                dist[v] = dist[u] + graph[u][v];
            }
        }
    }
}

int choose_best_root(int n, int graph[MAX_NODES][MAX_NODES]) {
    int dist[MAX_NODES];
    int best_root = -1;
    long long best_total = LLONG_MAX;

    for (int src = 0; src < n; src++) {
        dijkstra(n, graph, src, dist);

        long long total = 0;
        int reachable = 1;

        for (int i = 0; i < n; i++) {
            if (dist[i] >= INF) {
                reachable = 0;
                break;
            }
            total += dist[i];
        }

        if (reachable && total < best_total) {
            best_total = total;
            best_root = src;
        }
    }

    return best_root;
}`;

document.getElementById("cCode").textContent = cCode;

document.getElementById("optimizeBtn").addEventListener("click", () => {
  const n = Number(document.getElementById("nodeCount").value);
  const rawEdges = document.getElementById("edges").value.trim();
  const result = document.getElementById("result");

  if (!Number.isInteger(n) || n < 2 || n > 100) {
    result.textContent = "Node count must be an integer from 2 to 100.";
    return;
  }

  const graph = Array.from({ length: n }, () => Array(n).fill(Infinity));
  for (let i = 0; i < n; i++) graph[i][i] = 0;

  const lines = rawEdges.split("\n").map((line) => line.trim()).filter(Boolean);

  for (const line of lines) {
    const [a, b, w] = line.split(/\s+/).map(Number);
    if ([a, b, w].some((x) => Number.isNaN(x))) {
      result.textContent = `Invalid edge format: "${line}"`;
      return;
    }
    if (a < 0 || a >= n || b < 0 || b >= n || w < 0) {
      result.textContent = `Edge out of range or negative cost: "${line}"`;
      return;
    }

    graph[a][b] = Math.min(graph[a][b], w);
    graph[b][a] = Math.min(graph[b][a], w);
  }

  const dijkstra = (src) => {
    const dist = Array(n).fill(Infinity);
    const visited = Array(n).fill(false);
    dist[src] = 0;

    for (let step = 0; step < n; step++) {
      let u = -1;
      let best = Infinity;

      for (let i = 0; i < n; i++) {
        if (!visited[i] && dist[i] < best) {
          best = dist[i];
          u = i;
        }
      }

      if (u === -1) break;
      visited[u] = true;

      for (let v = 0; v < n; v++) {
        if (!visited[v] && graph[u][v] < Infinity && dist[u] + graph[u][v] < dist[v]) {
          dist[v] = dist[u] + graph[u][v];
        }
      }
    }

    return dist;
  };

  let bestRoot = -1;
  let bestTotal = Infinity;
  const details = [];

  for (let src = 0; src < n; src++) {
    const dist = dijkstra(src);

    if (dist.some((d) => !Number.isFinite(d))) {
      details.push(`Node ${src}: disconnected`);
      continue;
    }

    const total = dist.reduce((sum, d) => sum + d, 0);
    details.push(`Node ${src}: total cost = ${total}`);

    if (total < bestTotal) {
      bestTotal = total;
      bestRoot = src;
    }
  }

  if (bestRoot === -1) {
    result.innerHTML = "No valid root found (graph is disconnected).";
    return;
  }

  result.innerHTML = `<strong>Optimal root: Node ${bestRoot}</strong><br/>Total transfer cost: ${bestTotal}<br/><br/>${details.join("<br/>")}`;
});
