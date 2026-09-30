
## Reasoning

Reasoning is a class of AI algorithms that are centered around:
- Logical Language: Representing knowledge in a formalized way
- Inference: deriving new information from knowledge
- Knowledge: What I currently know to be true
- Goal: A theorem that we want to prove

## Propositional Logic

Propositions
- Logical symbols which we ascribe a certain meaning to.
- The computer doesn't know the meaning, It just knows the symble

Logical Connectives
^ and
 (insert symbol) or
 (insert symbol) not
 -> implication
 <-> bicondition

## Forward Search
start with a knoeldge base
follow infrence rules, add inferred knoledge
repeat until T or not T i si knowledge base

whast the sates space?
very similart to planning
differences: 
- answering unknown is hars
- inference rules not applicalble in all states
- Order of inference usually does not matter

This is an exaplle of an expliding state space, it is not efficient and we might end up with a state there are not possible and infinite. 