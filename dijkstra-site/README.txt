=====================================================
 Smart Paste Collection Root Optimizer
 Dijkstra's Algorithm — C + HTML/CSS/JS
=====================================================

FILES
-----
  dijkstra.c   — Full C source code (compile & run in terminal)
  index.html   — Main website page
  style.css    — Stylesheet (dark-themed, responsive)
  app.js       — JavaScript: Dijkstra logic + graph visualisation

HOW TO OPEN THE WEBSITE
-----------------------
  Just double-click index.html in any modern browser.
  No server, no build step, no dependencies needed.

HOW TO COMPILE & RUN THE C PROGRAM
------------------------------------
  Requirements: GCC (any version), Linux/macOS/Windows (MinGW)

  Compile:
    gcc -o dijkstra dijkstra.c -lm

  Run:
    ./dijkstra          (Linux/macOS)
    dijkstra.exe        (Windows)

  Follow the interactive prompts:
    1. Enter number of collection points (nodes)
    2. Enter number of connections (edges)
    3. Enter each edge as: source destination weight
    4. Choose mode:
       1 = Auto-find the optimal root
       2 = Run Dijkstra from a specific root you choose

ALGORITHM
---------
  Dijkstra's Single-Source Shortest Path (SSSP) is applied from
  every node. The root with the lowest total-cost (sum of all
  shortest paths to all other nodes) and maximum reachability
  is declared the Optimal Paste Collection Root.

  Time complexity: O(V * (V+E) log V)
  Space complexity: O(V + E)

WEBSITE FEATURES
----------------
  - Interactive graph builder (add nodes and weighted edges)
  - Visual canvas showing the network and highlighted optimal paths
  - Auto mode: optimizer picks the best root automatically
  - Manual mode: run Dijkstra from any node you choose
  - Shortest-path table for every destination node
  - All-roots comparison table with ranking
  - Embedded C source code viewer (click "Show Code")
  - Example network pre-loaded for quick demo

=====================================================
