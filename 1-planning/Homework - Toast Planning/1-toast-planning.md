# Toast Planning

**Problem Description:** Given a toaster, a plug, a plate and a slice of bread, and we want to end up with toasted bread sitting on the plate if possible in the quickest amount of time. There are six actions available and each one moves a clock forward, but they don't all cost the same: waiting takes 10 minutes while everything else takes 1.

**Area of AI:** Planning

**Applied Algorithms:**
- BFS (Breadth-first search): Take the oldest node on the frontier. The goal is checked when a node comes off the queue rather than when it goes on.
- Dijkstra: Take the node with the smallest cumulative time. A state is marked as done when it is popped, because by then we already know the cheapest way of getting there.
- RS (Random search): Was provided as an example.

Both of my searches identify a state by its four attributes and deliberately leave out the clock. The clock only ever goes up, so if I included it every state would look brand new and the search would never notice it was going in circles.

**Results:**
- BFS: Always gives the plan with the fewest actions, somewhere between 3 and 6 depending on where you start, but that is not the same as the fastest plan. It uses `wait` to turn the toaster off because that is only one action, even though it burns 10 minutes.

- Dijkstra: Gives the fastest plan instead. On the two hardest start states it works out that unplugging the toaster and plugging it back in turns it off in 2 minutes rather than 10, which brings the total down to 15 minutes where BFS needs 23 and 24. On the other two start states the two algorithms agree, because there the shortest plan happens to be the quickest one as well.

- Both only expand 9 or 10 states. Dropping the closed set makes the search run forever. There are only 16 different states in the whole problem, but you can wander between them for as long as you like, so there are endless paths through those 16.

**Location:** `1-planning/Homework - Toast Planning/` — the implementation is `breadth_first_search()`
and `shortest_toast_time_search()` in `planning/planner.py`.
