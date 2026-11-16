# SE_14 — Artificial Intelligence Basics

Coursework for module SE_14

## Repository structure

```
0-intro/                     
1-planning/                  
2-optimization/              
3-machine-learning/          
4-Reasoning/                 
5-large-language-models/     
_assets/                     
```

Inside each unit:

| path | what it holds |
|---|---|
| `Input/` | lecture slides as PDF, plus `class-notes.md` — my notes for that unit |
| `Homework - <name>/` | one folder per practical exercise |

And inside each homework folder:

| file | what it is |
|---|---|
| `EXERCISE.md` | the brief as handed out — what the task was |
| `README.md` | how to set it up and run it, and where my solution lives |
| `<unit>-<name>.md` | the high-level description submitted for the assessment |

## Homework

| unit | homework | area of AI | algorithms implemented | hand-in | how to run |
|---|---|---|---|---|---|
| 1 | [Maze Planning](1-planning/Homework%20-%20Maze%20Planning) | Planning | depth-first, breadth-first, heuristic search | [description](1-planning/Homework%20-%20Maze%20Planning/1-maze-planning.md) | [README](1-planning/Homework%20-%20Maze%20Planning/README.md) |
| 1 | [Toast Planning](1-planning/Homework%20-%20Toast%20Planning) | Planning | breadth-first search, uniform-cost search (Dijkstra) | [description](1-planning/Homework%20-%20Toast%20Planning/1-toast-planning.md) | [README](1-planning/Homework%20-%20Toast%20Planning/README.md) |
| 4 | [Knights and Knaves](4-Reasoning/Homework%20-%20Knigts%20and%20Knaves) | Reasoning | propositional logic model checking | [description](4-Reasoning/Homework%20-%20Knigts%20and%20Knaves/4-knights-and-knaves.md) | [README](4-Reasoning/Homework%20-%20Knigts%20and%20Knaves/README.md) |

The repository also contains the exercises for optimization (`2-optimization/`) and machine
learning (`3-machine-learning/`). Those are coursework and are not part of this assessment
submission.

## Running the homework

Each homework folder has its own README with the detail. In short:

| homework | command |
|---|---|
| Maze Planning | `python main.py maze1.txt BFS` |
| Toast Planning | `python main.py` |
| Knights and Knaves | `python puzzle.py` |
