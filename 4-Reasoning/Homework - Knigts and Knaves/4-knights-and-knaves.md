# Knights and Knaves

**Problem Description:** All of the characters in these puzzles are either a knight – someone who always tells the truth, or a knave – someone who always lies. We are told what each one of them says and need to figure out who is who based on propositional logic.

**Area of AI:** Reasoning

**Applied Algorithms:**
- Model checking: Check all possible combinations of true or false for all the symbols involved, discard the combinations that go against our knowledge and see what is left.

Each knowledge base exercise includes two types of statements:

- Rules of the game: a character is a knight if and only if they are not a knave. If this is not included in the exercise, then the checker would have no way of knowing that a certain person cannot be both at the same time.
- Statement of what each character said, as a biconditional of being a knight and having said something true. If the statement is false, then the biconditional implies that the character in question is a knave.

**Results:**
- All four exercises are solved successfully.
    - Puzzle 0: A is a knave.
    - Puzzle 1: A is a knave and B is a knight. 
    - Puzzle 2: A is a knave and B is a knight. 
    - Puzzle 3: A is a knight, B is a knave and C is a knight.

- Writing `Biconditional(AKnight, <what A said>)` says everything: a knight's statement is true and a knave's is false, both directions at once, and the model checker takes it from there.

**Location:** `4-Reasoning/Homework - Knigts and Knaves/` — the implementation of the knowledge bases `knowledge0` through
`knowledge3` are in `puzzle.py`.
