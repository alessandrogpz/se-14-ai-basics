BFS always guarantees the shortest path, but does not guarantees the shortest execution time.

DFS does not guarantees the shortest path nor the shortest execution time

HS is a method of estimating the next move based on a value given (in a maze case, it can be the Euclidian distance between the difference of the goal and current position.), in this way prioritizing always the next move with a smaller number difference

Dijkstra is HS using path length
A* is HS using path length and Euclidian distance to the goal

![[maze_planning.png]]

Black numbers are path length
Orange numbers are Euclidian distance to final goal