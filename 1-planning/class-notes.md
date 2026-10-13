### BFS (Breadth First Search)
Breadth First Search always guarantees the shortest path, but does not guarantees the shortest execution time.

It works by always exploring all the nodes of a tree that are in the same level. 

It uses a FIFO queues as a data structure

### DFS (Depth First Search)
It does not guarantees the shortest path nor the shortest execution time

Differently than ***BFS***, ***DFS*** works by always exploring a path until it cannot anymore, only then moving to a layer above and the next node.  

It uses a LIFO queues as a data structure

### HS (Heuristic Search)
It is a method of estimating the next move based on a value given (in a maze case, it can be the Euclidean distance between the difference of the goal and current position.), in this way prioritizing always the next move with a smaller number difference

Algorithms like ***Dijkstra*** and **A*** belong to the *Heuristic Search* category

- **Dijkstra**
	It uses the cumulative path length as a way to choose the best nodes to explore at every iteration
	
	Dijkstra still uses a queue — a **priority queue** using *h* as a deciding factor. (where *h* is the cumulative length)
	
	***BFS*** _is_ **Dijkstra** for the special case where every edge costs the same.

- **A*** is a heuristic search algorithm using cumulative path length and Euclidean distance to the goal
	It combines the cumulative length and a weight to determine the best possible node. The weight in a maze can be the physical distance from the current node to the goal (using Euclidean distance). 
	
	**A*** still uses a queue — a **priority queue** using *h* + *g* as a deciding factor. (where *h* is the cumulative length and *g* is the weight chosen)

![[maze_planning.png]]

PS.: Black numbers are path length and orange numbers are Euclidean distance to final goal


| | frontier discipline | finds |
|---|---|---|
| DFS | stack — LIFO | *a* path |
| BFS | queue — FIFO | fewest edges |
| Dijkstra / UCS | priority queue by `g` | cheapest path |
| Greedy best-first | priority queue by `h` | fast, no guarantee |
| A* | priority queue by `g + h` | cheapest path, fewer expansions |
