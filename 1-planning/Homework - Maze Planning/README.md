# Maze Planning — How to Run

- [EXERCISE.md](EXERCISE.md)
- [1-maze-planning.md](1-maze-planning.md)

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Pass a maze file and a search algorithm:

```
python main.py maze1.txt BFS
```

Mazes `maze1.txt` through `maze7.txt` live in `mazes/` — pass just the filename, not the folder.

| arg | algorithm |
|---|---|
| `RS`  | random search (provided)
| `DFS` | depth-first search
| `BFS` | breadth-first search
| `HS`  | heuristic search
