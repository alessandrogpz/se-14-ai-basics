## Large Language Models (LLMs)

The question of this unit: how does an LLM-based chat program (like ChatGPT) take a textual request and produce a textual answer?

Two disclaimers:

- Many chat systems (especially ChatGPT) are closed or only partially open (e.g., open weights), so parts of this are **educated guesses**.
- The concepts are **simplified** to keep them digestible.

The topic is split in two parts:

1. **Chat** — assume the LLM is just a next-word predictor and look at the program built around it.
2. **GPT** — look inside the LLM: how text is represented, how the model is structured and how it is trained.

---

## Part 1: The Chat Program

### High Level Workflow

Think of the LLM as the next-word predictor of a smartphone keyboard. The chat program around it works like this:

1. The user enters a question, which is stored in the **chat history**.
2. The answer is produced **word by word** (we say *token* by token), repeating three steps:
   1. **Predict Next Token** — ask the LLM for probabilities of the next token
   2. **Choose Next Token** — pick one token based on those probabilities
   3. **Interpret Answer** — add the token to the answer and the chat history; this step may also call external tools (APIs)

The LLM has two inputs: the **system prompt** and the **chat history**.

### System Prompt

It exists before any user input is processed. LLMs are word prediction machines: they can complete any sentence and play any role, so they have to be told how to behave.

- **Conversation context:** tells the model its name and that it is chatting with a user via text
- **Role:** tells the model how the role it plays should be perceived, which helps it predict words that fit that role
- **Tools:** describes which tools exist and how to access them

The example in the slides is a leaked ChatGPT 4.5 system prompt. It was leaked through ChatGPT itself, so it may contain omissions and hallucinations. System prompts vary between models; a collection of known ones: https://github.com/elder-plinius/CL4R1T4S

### Tools in the System Prompt

The input and output of an LLM is **only text**, so the LLM cannot call a tool itself. Instead:

- Its textual output includes instructions that make the **surrounding system** call the tool.
- If the tool returns something, the result is put back into the **text input** of the model.

Examples:

- **bio** → lets the model remember information beyond the current conversation. The model addresses its message to `bio` instead of the user. The message is not shown in the chat window; it is stored separately and added to every future conversation.
- **dalle** → image generation. The model does not generate the image; it writes a prompt in a specific format, and the surrounding code picks it out, calls Dalle and inserts the image into the answer.

### Chat History

The chat history is not just a log of the chat. It is **the text that the LLM auto-completes**, similar to writing the beginning of a dialog in a book and asking for the next words.

The LLM does **not remember** anything between predictions. For every single next token it looks at the full history again.

So the chat history has two purposes:

- The chat system remembers the complete conversation
- It is the text given to the LLM to complete

Besides the current chat it probably also contains:

- Information remembered from past conversations (likely how `bio` memories are integrated)
- The **partially generated answer** — since the answer is generated one token at a time, the model keeps feeding itself what it has already written

### Predict Next Token

Internal workflow:

1. **Compile input** — combine system prompt and chat history. The easiest way is to concatenate them, but the LLM has a limited input length (**context window**), so the chat history is probably abbreviated to fit.
2. **Call LLM** — the output is a list of **token scores**. A high score means the token is very likely to continue the conversation, a low score means it is very unlikely.
3. **Create distribution** — the scores are normalized into a probability distribution (they add up to 1).

**Temperature** controls how "peaked" the distribution becomes:

| Temperature | Effect |
|---|---|
| Low | most of the probability goes to the highest scores |
| High | more uniform distribution |

High temperature is often associated with "creativeness", but more randomness is not necessarily the same as creativity.

### Choose Next Token

- **The obvious option:** pick a token according to its probability. This may be all there is.
- **Could there be more?** LLMs produce structured text like XML or JSON (e.g., tool calls use JSON). Parsing and input validation could be used here to auto-correct mistakes made by the LLM.

### Interpret Answer

- **Most of the time:** the token is forwarded to the **chat window** (shown to the user) and to the **chat history** (part of the prompt for the next LLM call).
- **Sometimes:** call an API instead. The interpreter constantly parses the produced text for tool keywords; when it finds one, it parses the following text until it has enough information to make the function call.

Tools mentioned in the leaked ChatGPT 4.5 system prompt:

| Tool | What it does |
|---|---|
| **canmore** | document editing window; the model can open and edit documents |
| **dalle** | the model writes a prompt, Dalle generates the image shown to the user |
| **python** | Jupyter environment to run code for calculations or visualizations |
| **web** | search engine calls; results are added to the chat history for later predictions |
| **bio** | long-term memory |

### MCP (Model Context Protocol)

A protocol to make **new tools** available to a chat model: you host the functionality on a server and provide a client that connects it to the LLM. Probably integrated the same way as the other tools:

- A standardized text that describes to the model how to use the tool (same level as the system prompt)
- A textual pattern the model uses to call the tool and hand over data
- A parser that looks for this pattern, forwards the call and processes the result

---

## Part 2: GPT

### Text Completion Problem - Formalized

The LLM is a machine learning model, so we can define the AI problem it solves:

- **Input:** `Text: Token*`
- **Output:** `scores: Token → Float`
- **Success criteria:** the scores represent the correctness of the text continuation
- **Quality criteria:** prediction in under 1 second

Legend used on the board from the discussion session:

| Letter | Meaning |
|---|---|
| **T** | number of tokens |
| **E** | number of embedding dimensions |
| **W** | number of words |
| **V** | number of tokens in the vocabulary |
| **B** | number of transformer blocks |
| **A** | number of attention heads |
| **F** | number of feed forward dimensions |

### 1) How is text represented in an LLM?

![[Screenshot_20261013_152157.png]]

The text has to become a matrix of numbers: one row per token, one column per embedding dimension → `float[T][E]`. This happens in two steps.

**Tokenization** (text → token IDs):

| Step | From → To | What happens |
|---|---|---|
| **regex** | `String → String[W]` | split the text into words |
| **UTF-8** | `String[W] → Bytes[W][]` | turn every word into its bytes |
| **Byte-Pair Encoding** | `Bytes[W][] → Int[W][]` | merge the bytes of each word into token IDs |
| **Flatten** | `Int[W][] → Int[T]` | one flat list of token IDs |

A token is not always a whole word, and each token is one integer ID from the vocabulary:

| Hey | G | ippi | ty | , | what | do | you | think | I | should | have | for | dinner | ? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 19182 | 480 | 25240 | 1919 | 11 | 1148 | 656 | 499 | 1781 | 358 | 1288 | 617 | 369 | 14177 | 30 |

**Embedding** (token IDs → vectors):

| Step | From → To | What happens |
|---|---|---|
| **Embedding Table Lookup** | `Int[T] → float[T][E]` | every token ID is replaced by a vector of `E` numbers |
| **Positional Encoding** | `float[T][E] → float[T][E]` | adds the position of the token in the text |

The **embedding space** is the space the LLM reasons in: directions carry meaning, e.g. `E(queen) - E(king) ≈ E(woman) - E(man)`.

### 2) How is an LLM structured?

The model turns the embeddings `float[T][E]` into one score per vocabulary token: `float[V]`.

In between is the **transformer block**, repeated **B times**:

`Normalization → Masked Multi Self Attention → ⊕ → Normalization → Feed Forward → ⊕`

The output of the last block is still `float[T][E]`; a final **Normalization** and **Feed Forward** step turns it into the scores `float[V]`.

- **LayerNorm**
	Normalizes the embedding of each token.

- **Masked Multi Self Attention** (**A** heads)
	Every token looks at the other tokens to update its own meaning, using three values:

	| | Question it answers |
	|---|---|
	| **K: Key** | Which type of token am I? |
	| **Q: Query** | Which types of token do I attend to? |
	| **V: Value** | How do I update my embedding? |

	$$
	\text{softmax}\left(\frac{K^T Q}{\sqrt{d_k}}\right) V
	$$

- **Feed Forward** (applied **T** times, once per token)
	`Fully Connected Layer (F neurons) → ReLU → Fully Connected Layer (E neurons)`
	- First layer → extract interesting features
	- ReLU → non-linearity
	- Second layer → calculate feature effects

- **⊕ Skip Connection**
	`old meaning float[T][E] + meaning update float[T][E] = new meaning float[T][E]`
	Attention and feed forward do not replace the embedding, they only add an update to it.

### 3) How are LLMs trained?

![[Screenshot_20261013_152222.png]]

Three stages: **Pretraining → Finetuning → Reinforcement Learning from Human Feedback**

| | Pretraining | Finetuning | RLHF |
|---|---|---|---|
| **Type** | Supervised Learning | Supervised Learning | Reinforcement Learning |
| **Dataset** | "a good chunk of the internet" | thousands of Prompt → Answer examples | LLM dialogs rated by humans |
| **Result** | general purpose auto-complete | prompt answering | better answers according to policies |

- **Pretraining**
	- **Dataset:** "a good chunk of the internet" (Andrej Karpathy), scraped from web pages, video captions, books, ...
	- **GPT-3** (according to the internet):
		- 570 GB of compressed text (45 TB before filtering)
		- ~102,000,000,000 words (if uncompressed)
		- ~212,511 Lord of the Rings trilogies
		- Average text compression rates between 2 and 4
	- **Training:** the usual backpropagation, based on Gradient Descent (e.g., Adam optimizer)
	- **Optimizations:**
		- **Masking:** during the attention blocks, later tokens are excluded from attention. This makes it possible to train multiple samples from one context window.
		- **Skip Connections:** combat vanishing gradients
		- **Layer Normalization:** improved training speed
	- **Result:** a general purpose auto-complete. Not specifically good at conversations, and it will not understand chat-specific instructions (e.g., tools).

- **Finetuning**
	- **Dataset:** specific examples of Prompt → Answer. Originally generated by humans, today probably also partially by LLMs. On the order of thousands (Andrej Karpathy).
	- **Training:** starts with the weights from the previous step, otherwise the procedure remains the same.
	- **Result:** specialized in answering prompts, but may be unclear, offensive, unhelpful, ... (it's learned from the internet after all).

- **Reinforcement Learning from Human Feedback**
	- **Value Model:** another machine learning model, trained via Supervised Learning. Its dataset is example dialogs from the LLM with human-generated ratings of the responses.
	- **Training:** Proximal Policy Optimization (PPO) = Deep Reinforcement Learning
	- **Result:** improves the quality of answers according to policies (depends on how the humans were instructed to rate the responses).
