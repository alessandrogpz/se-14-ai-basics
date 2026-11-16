# Maze Planning

**Problem Description:** Given a maze with an entry and exit node, our goal is to reach the exit node from the entry node and evaluate the steps taken to do so.

**Area of AI:** Planning

**Applied Algorithms:** 
- DFS (Depth-first search): Take the newest node on the frontier.
- BFS (Breadth-first search): Take the oldest node on the frontier.
- HS (Heuristic / greedy best-first search): Take the node with the smallest
Euclidean distance to the goal. 
- RS (Random Search): Search was provided as an example.

**Results:** 
- DFS: It can get lucky and solve the problem quickly but its prone to taking visible detours.

- BFS: It finds the shortest path at the expense of searching every single node that has an equal or shorter lenght distance difference from the current to the exit node.

- HS: Euclidian distance is measured through walls. In maze 4, it performs the shortest possible path in the least ammount of calculations possible, since there are no walls in between the entry and exit node, but this behaviour is not guaranteed. 

**Location:** `1-planning/Homework - Maze Planning/` — the implementation is `select_node()` in
[`maze_solver/frontier.py`](maze_solver/frontier.py).