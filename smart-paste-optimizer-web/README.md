# Smart Paste Collection Root Optimizer

This mini-project provides:
1. A browser UI to test weighted-graph input and choose the best collection root.
2. A C implementation of Dijkstra-based root optimization.

## Requirements

### Runtime requirements
- **Web UI**: Any modern browser (Chrome, Edge, Firefox, Safari).
- **C compiler**: GCC or Clang with C99 support.

### System requirements
- OS: Linux, macOS, or Windows.
- RAM: 256 MB minimum.
- Disk: < 10 MB for source and build artifacts.

## Build and run (C)

```bash
gcc dijkstra.c -O2 -std=c99 -Wall -Wextra -o dijkstra
./dijkstra
```

Expected output (for the included sample graph):

```text
Optimal root: 1
Total cost: 26
```

## How the optimizer works
- For each node, Dijkstra computes shortest paths to all other nodes.
- The node with the lowest sum of shortest-path distances is selected as the optimal root.
- If any node is unreachable from a candidate, that candidate is rejected.

## Web UI usage
1. Open `index.html` in a browser.
2. Set number of nodes.
3. Enter edges in `from to cost` format (one edge per line).
4. Click **Optimize Root**.

## Notes
- Current UI assumes an undirected graph and non-negative edge costs.
- For directed networks, remove mirrored edge assignment in `script.js` and `dijkstra.c`.
