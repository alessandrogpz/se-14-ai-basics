## Reasoning

Reasoning is a class of AI algorithms that are centered around:

- **Logical Language:** representing knowledge in a formalized way
- **Inference:** deriving new information from knowledge
- **Knowledge:** what I currently know to be true
- **Goal:** a theorem that we want to prove

## Propositional Logic

### Propositions

- Logical symbols which we ascribe a certain meaning to.
- The computer doesn't know the meaning, it just knows the symbol.

### Logical Connectives

| Symbol | Meaning |
|---|---|
| ∧ | and |
| ∨ | or |
| ¬ | not |
| → | implication |
| ↔ | biconditional |

## Forward Search

1. Start with a knowledge base
2. Follow inference rules, add the inferred knowledge
3. Repeat until *T* or ¬*T* is in the knowledge base

### What's the state space?

Very similar to ***planning***. Differences:

- Answering "unknown" is hard
- Inference rules are not applicable in all states
- Order of inference usually does not matter

This is an example of an **exploding state space**: it is not efficient, and we might end up with states that are not possible and with an infinite state space.
