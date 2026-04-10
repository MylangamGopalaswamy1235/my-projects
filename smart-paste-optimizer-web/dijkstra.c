#include <stdio.h>
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

int choose_best_root(int n, int graph[MAX_NODES][MAX_NODES], long long *best_total) {
    int dist[MAX_NODES];
    int best_root = -1;
    *best_total = LLONG_MAX;

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

        if (reachable && total < *best_total) {
            *best_total = total;
            best_root = src;
        }
    }

    return best_root;
}

int main(void) {
    int n = 6;
    int graph[MAX_NODES][MAX_NODES];

    for (int i = 0; i < MAX_NODES; i++) {
        for (int j = 0; j < MAX_NODES; j++) {
            graph[i][j] = (i == j) ? 0 : INF;
        }
    }

    int edges[][3] = {
        {0, 1, 4}, {0, 2, 2}, {1, 2, 1}, {1, 3, 5},
        {2, 3, 8}, {2, 4, 10}, {3, 4, 2}, {3, 5, 6}, {4, 5, 3}
    };

    int m = sizeof(edges) / sizeof(edges[0]);
    for (int i = 0; i < m; i++) {
        int u = edges[i][0], v = edges[i][1], w = edges[i][2];
        graph[u][v] = w;
        graph[v][u] = w;
    }

    long long best_total;
    int root = choose_best_root(n, graph, &best_total);

    if (root == -1) {
        printf("No valid root found (disconnected graph).\n");
    } else {
        printf("Optimal root: %d\n", root);
        printf("Total cost: %lld\n", best_total);
    }

    return 0;
}
