# Knights and Knaves

**Problem Description:** Every character in these puzzles is either a knight, who always tells the truth, or a knave, who always lies. We are told what each of them says and have to work out who is which. The point of the exercise is to write down what we know in propositional logic so that a model checking algorithm can do the reasoning instead.

**Area of AI:** Reasoning

**Applied Algorithms:**
- Model checking: Go through every possible combination of true and false for all the symbols, throw away the combinations that contradict what we know, and see what is left.

The algorithm itself was provided. The work was in writing the knowledge bases, and each one has two kinds of statement in it:
- The rules of the game, which are the same every time: a character is a knight exactly when they are not a knave. Without this the checker has no reason to rule out somebody being both at once.
- What each character said, written as a biconditional between that character being a knight and their statement being true. If the statement is false then the biconditional forces the speaker to be a knave.

**Results:**
- All four puzzles come out correctly. 
    - Puzzle 0: A is a knave. 
    - Puzzle 1: A is a knave and B is a knight. 
    - Puzzle 2: A is a knave and B is a knight. 
    - Puzzle 3: A is a knight, B is a knave and C is a knight.

- Writing `Biconditional(AKnight, <what A said>)` says everything: a knight's statement is true and a knave's is false, both directions at once, and the model checker takes it from there.

**Location:** `4-Reasoning/Homework - Knigts and Knaves/` — the implementation of the knowledge bases `knowledge0` through
`knowledge3` are in `puzzle.py`.
