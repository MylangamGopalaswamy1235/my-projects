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

typedef struct MinHeapNode {
    int vertex;
    int dist;
} MinHeapNode;

typedef struct MinHeap {
    int size;
    int capacity;
    int* pos;
    MinHeapNode** array;
} MinHeap;

Graph* createGraph(int numNodes) {
    Graph* graph = (Graph*)malloc(sizeof(Graph));
    graph->numNodes = numNodes;
    for (int i = 0; i < numNodes; i++)
        graph->adjList[i] = NULL;
    return graph;
}

void addEdge(Graph* graph, int src, int dest, int weight) {
    Edge* edge = (Edge*)malloc(sizeof(Edge));
    edge->dest = dest;
    edge->weight = weight;
    edge->next = graph->adjList[src];
    graph->adjList[src] = edge;

    Edge* revEdge = (Edge*)malloc(sizeof(Edge));
    revEdge->dest = src;
    revEdge->weight = weight;
    revEdge->next = graph->adjList[dest];
    graph->adjList[dest] = revEdge;
}

MinHeapNode* newMinHeapNode(int v, int dist) {
    MinHeapNode* node = (MinHeapNode*)malloc(sizeof(MinHeapNode));
    node->vertex = v;
    node->dist = dist;
    return node;
}

MinHeap* createMinHeap(int capacity) {
    MinHeap* heap = (MinHeap*)malloc(sizeof(MinHeap));
    heap->pos = (int*)malloc(capacity * sizeof(int));
    heap->size = 0;
    heap->capacity = capacity;
    heap->array = (MinHeapNode**)malloc(capacity * sizeof(MinHeapNode*));
    return heap;
}

void swapMinHeapNode(MinHeapNode** a, MinHeapNode** b) {
    MinHeapNode* t = *a;
    *a = *b;
    *b = t;
}

void minHeapify(MinHeap* heap, int idx) {
    int smallest = idx;
    int left = 2 * idx + 1;
    int right = 2 * idx + 2;

    if (left < heap->size && heap->array[left]->dist < heap->array[smallest]->dist)
        smallest = left;
    if (right < heap->size && heap->array[right]->dist < heap->array[smallest]->dist)
        smallest = right;
    if (smallest != idx) {
        MinHeapNode* smallestNode = heap->array[smallest];
        MinHeapNode* idxNode = heap->array[idx];
        heap->pos[smallestNode->vertex] = idx;
        heap->pos[idxNode->vertex] = smallest;
        swapMinHeapNode(&heap->array[smallest], &heap->array[idx]);
        minHeapify(heap, smallest);
    }
}

int isEmpty(MinHeap* heap) {
    return heap->size == 0;
}

MinHeapNode* extractMin(MinHeap* heap) {
    if (isEmpty(heap)) return NULL;
    MinHeapNode* root = heap->array[0];
    MinHeapNode* lastNode = heap->array[heap->size - 1];
    heap->array[0] = lastNode;
    heap->pos[root->vertex] = heap->size - 1;
    heap->pos[lastNode->vertex] = 0;
    --heap->size;
    minHeapify(heap, 0);
    return root;
}

void decreaseKey(MinHeap* heap, int v, int dist) {
    int i = heap->pos[v];
    heap->array[i]->dist = dist;
    while (i && heap->array[i]->dist < heap->array[(i - 1) / 2]->dist) {
        heap->pos[heap->array[i]->vertex] = (i - 1) / 2;
        heap->pos[heap->array[(i - 1) / 2]->vertex] = i;
        swapMinHeapNode(&heap->array[i], &heap->array[(i - 1) / 2]);
        i = (i - 1) / 2;
    }
}

int isInMinHeap(MinHeap* heap, int v) {
    return heap->pos[v] < heap->size;
}

typedef struct Result {
    int dist[MAX_NODES];
    int parent[MAX_NODES];
    int totalCost;
    int reachable;
} Result;

Result dijkstra(Graph* graph, int src) {
    int V = graph->numNodes;
    Result result;
    result.totalCost = 0;
    result.reachable = 0;

    int* dist = result.dist;
    int* parent = result.parent;

    MinHeap* heap = createMinHeap(V);

    for (int v = 0; v < V; v++) {
        dist[v] = INF;
        parent[v] = -1;
        heap->array[v] = newMinHeapNode(v, dist[v]);
        heap->pos[v] = v;
    }

    dist[src] = 0;
    heap->array[src] = newMinHeapNode(src, dist[src]);
    heap->pos[src] = src;
    decreaseKey(heap, src, dist[src]);
    heap->size = V;

    while (!isEmpty(heap)) {
        MinHeapNode* u = extractMin(heap);
        int uVertex = u->vertex;
        free(u);

        Edge* adj = graph->adjList[uVertex];
        while (adj != NULL) {
            int v = adj->dest;
            if (isInMinHeap(heap, v) && dist[uVertex] != INF &&
                adj->weight + dist[uVertex] < dist[v]) {
                dist[v] = dist[uVertex] + adj->weight;
                parent[v] = uVertex;
                decreaseKey(heap, v, dist[v]);
            }
            adj = adj->next;
        }
    }

    for (int i = 0; i < V; i++) {
        if (i != src && dist[i] != INF) {
            result.totalCost += dist[i];
            result.reachable++;
        }
    }

    free(heap->pos);
    for (int i = 0; i < V; i++) {
        if (heap->array[i]) free(heap->array[i]);
    }
    free(heap->array);
    free(heap);

    return result;
}

int findOptimalRoot(Graph* graph) {
    int V = graph->numNodes;
    int bestRoot = 0;
    int bestCost = INT_MAX;

    for (int r = 0; r < V; r++) {
        Result res = dijkstra(graph, r);
        if (res.reachable == V - 1 && res.totalCost < bestCost) {
            bestCost = res.totalCost;
            bestRoot = r;
        }
    }
    return bestRoot;
}

void printPath(int* parent, int j) {
    if (parent[j] == -1) {
        printf("%d", j);
        return;
    }
    printPath(parent, parent[j]);
    printf(" -> %d", j);
}

void freeGraph(Graph* graph) {
    for (int i = 0; i < graph->numNodes; i++) {
        Edge* edge = graph->adjList[i];
        while (edge) {
            Edge* next = edge->next;
            free(edge);
            edge = next;
        }
    }
    free(graph);
}

int main() {
    int V, E;
    printf("=== Smart Paste Collection Root Optimizer ===\n");
    printf("Using Dijkstra's Algorithm\n\n");

    printf("Enter number of collection points (nodes): ");
    scanf("%d", &V);
    if (V <= 0 || V > MAX_NODES) {
        printf("Error: nodes must be between 1 and %d\n", MAX_NODES);
        return 1;
    }

    printf("Enter number of connections (edges): ");
    scanf("%d", &E);
    if (E < 0) {
        printf("Error: edges cannot be negative\n");
        return 1;
    }

    Graph* graph = createGraph(V);

    printf("\nEnter edges (source destination weight):\n");
    for (int i = 0; i < E; i++) {
        int src, dest, weight;
        printf("Edge %d: ", i + 1);
        scanf("%d %d %d", &src, &dest, &weight);
        if (src < 0 || src >= V || dest < 0 || dest >= V || weight < 0) {
            printf("Invalid edge. Nodes 0..%d, weight >= 0\n", V - 1);
            i--;
            continue;
        }
        addEdge(graph, src, dest, weight);
    }

    printf("\nChoose mode:\n");
    printf("  1 - Find optimal root automatically\n");
    printf("  2 - Run Dijkstra from a specific root\n");
    printf("Choice: ");
    int mode;
    scanf("%d", &mode);

    if (mode == 1) {
        printf("\nAnalyzing all possible roots...\n");
        int optRoot = findOptimalRoot(graph);
        Result res = dijkstra(graph, optRoot);
        printf("\n=== RESULT ===\n");
        printf("Optimal root: Node %d\n", optRoot);
        printf("Total collection cost (sum of shortest paths): %d\n", res.totalCost);
        printf("Nodes reachable: %d / %d\n\n", res.reachable, V - 1);
        printf("Shortest paths from root %d:\n", optRoot);
        for (int i = 0; i < V; i++) {
            if (i == optRoot) continue;
            if (res.dist[i] == INF)
                printf("  Node %d: Unreachable\n", i);
            else {
                printf("  Node %d: distance=%d, path: ", i, res.dist[i]);
                printPath(res.parent, i);
                printf("\n");
            }
        }
    } else {
        int root;
        printf("Enter root node (0 to %d): ", V - 1);
        scanf("%d", &root);
        if (root < 0 || root >= V) {
            printf("Invalid root node.\n");
            freeGraph(graph);
            return 1;
        }
        Result res = dijkstra(graph, root);
        printf("\n=== RESULT ===\n");
        printf("Root: Node %d\n", root);
        printf("Total collection cost: %d\n", res.totalCost);
        printf("Nodes reachable: %d / %d\n\n", res.reachable, V - 1);
        printf("Shortest paths from root %d:\n", root);
        for (int i = 0; i < V; i++) {
            if (i == root) continue;
            if (res.dist[i] == INF)
                printf("  Node %d: Unreachable\n", i);
            else {
                printf("  Node %d: distance=%d, path: ", i, res.dist[i]);
                printPath(res.parent, i);
                printf("\n");
            }
        }
    }

    freeGraph(graph);
    return 0;
}
