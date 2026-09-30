## Learning in Machine Learning Is Statistical

Machine learning (in a general sense) is deriving a function from examples. The more examples of the results we have, the better we can approximate a function that covers them.

Formally, machine learning is an automated process for learning a function from data samples:

$$
\texttt{fun}: I_1 \times I_2 \times \dots \times I_n \rightarrow O
$$

- $I_1 \dots I_n$ → the inputs of the function
- $O$ → the output of the function

Once the learning is finished we can *use* that function: predict an output for a new input, or find the input that gives us the output we want.

### It's the Toaster Again

None of us know exactly how a toaster behaves at a given setting. But after a few tries we get a feeling for it — we have estimated the function behind toasting purely from our own observations. Same for gravity: we all knew things fall down long before we heard of Newton.

![[toaster-learning.png]]

When we have information about both the input and the output, it is called supervised learning.

### Types of Learning

- **Supervised Learning**: learns from samples of input and output — $(i_1, \dots, i_n, o)$.
- **Unsupervised Learning**: learns from samples of input only — $(i_1, \dots, i_n)$.
- **Reinforcement Learning**: no direct output is available, only a reinforcement signal — $(i_1, \dots, i_n, o, r)$.
  - e.g., moves in chess, which can only be judged good or bad after the whole game is finished.
  - The goal is to learn a function that **maximises** $r$.

The three categories split further:

| Type | Sub-problems |
|---|---|
| **Supervised** | **Regression** ($O$ is continuous → guess a number) · **Classification** ($O$ is discrete → pick a class) |
| **Unsupervised** | Clustering · Anomaly detection · Dimensionality reduction |
| **Reinforcement** | Learning the reward vs. learning action selection directly · **active** vs. **passive** learning |

**Active vs. passive learning** can apply to all three types: an *active* learner can influence which samples it gets (e.g., an agent deliberately playing a suboptimal move to explore a state it has never seen), a *passive* one just takes what it is given.

---

## Supervised Learning

Formalised, the problem is:

- **Input**: a set of samples of input and output of our hidden function — $(i_1^1, \dots, i_m^1, o^1), \dots, (i_1^n, \dots, i_m^n, o^n)$
- **Output**: a function $f_{learned}: I_1 \times \dots \times I_m \rightarrow O$
- **Success criteria**: $f_{learned}$ minimises some loss function comparing it to the samples

**Which type of function do we choose?**

The state space has two parts, and they are handled by completely different mechanisms:

1. **The model** — which *type* of function explains the samples best? Linear, curvy, sine-shaped…? → chosen semi-manually by the **machine learning expert**.
2. **The model parameters** — how to configure that function, e.g. $a$ and $b$ in $f(x) = ax + b$ → found by a **machine learning algorithm**.

The number of options grows with the number of parameters and the options per parameter. It is the same exploding state space we met in optimization.

### Some Regression Models

![[regression-models.png]]

| Model | Function |
|---|---|
| Linear Regression | $f(x) = ax + b$ |
| Polynomial Regression | $f(x) = ax^2 + bx + c$ |
| Another polynomial Regression | $f(x) = ax^3 + bx^2 + cx + d$ |
| Logistic Regression | $f(x) = \dfrac{a}{1 + e^{-b(x-c)}}$ |

Each of these functions looks different. By choosing one, you make an **assumption about the shape of the hidden function**. How do you arrive at that assumption? Data science.

Neural networks are so popular because they can represent almost any function, no matter how complex — so you don't have to make that assumption. But the burden does not disappear, it just moves: the learning algorithm gets more complex, the result is harder to interpret, and you now have to configure the network itself (layers, neurons per layer, …).

When not using neural networks, the model is usually chosen by visualizing the data.

### The Loss Function

Training = adjusting the learnable parameters to maximise the similarity between $f_{learned}$ and the training data. The loss function is how we **judge the quality of the current parameters**, and training tries to minimise it.

The sum of the absolute differences between each data point and the function line (measured vertically, at the same $x$), divided by the number of points:

$$
L = \frac{1}{n} \sum_{i=1}^{n} \left| y_i - f(x_i) \right|
$$

- $n$ → number of data points
- $x_i$ → input of the $i$-th data point
- $y_i$ → actual output of the $i$-th data point
- $f(x_i)$ → value predicted by the function (the point on the function line)
- $\left| y_i - f(x_i) \right|$ → absolute error of a single point

![[loss-function.png]]

Different problems use different loss functions:

- **Regression** → measure the *distance* between expected and actual value. Mean absolute error as above, or mean squared error:
  $$
  L_{MSE} = \frac{1}{n} \sum_{i=1}^{n} \left( y_i - f(x_i) \right)^2
  $$
  Squaring means **bigger errors are punished more** than smaller ones.
- **Classification** → measure the *probability* of guessing correctly, e.g. cross-entropy.

> Selecting parameters to minimise a function is something we've already done — it's **optimization**. That is why so many ML models use optimization algorithms for training. Neural networks use gradient descent.

### Data Quality

Data science and machine learning go hand in hand: before even starting the machine learning process, we need really good quality data. Otherwise, it's "Garbage In → Garbage Out".

Before you start learning anything, look at your data:

- Does it contain missing values or outliers?
- Are the data types correct?
- Is the data balanced?
- Are there patterns that help you select a model?

You will never eliminate all problems. Data can be:

- **Noisy** — no sensor measures with 100% precision (resolution, sensitivity, hardware error).
- **Biased** — a temperature sensor sitting in the shade of a tree reads low all afternoon. Or: if you only read it from November to February, you conclude Berlin is cold all year round. **Bias comes from having sparse or no data in parts of the state space.**

### Overfitting

When trying to fit every data point precisely, we risk overfitting the model. This means it becomes extremely specialized in that dataset and does not generalize well to new data — it has learned the inaccuracies inherent in the dataset.

![[overfitting.png]]

To detect it, set aside part of the data only for testing, e.g., train on 80% of the data and test on the remaining 20%.

**But overfitting can happen twice**, so we usually split three ways:

![[train-validation-test.png]]

- **Training set** — used during training.
- **Validation set** — estimates generalisation *during model selection*. Without it, we would tune the model against the test set and overfit the model-selection step too.
- **Test set** — estimates generalisation *after* model selection, touched at the very end.

---

## Machine Learning - Choices

Machine learning is different from the other areas of AI. Rather than selecting and adapting algorithms, it is mostly about **making good choices**:

**Data Preparation**
- Which columns to use
- How to preprocess them

**Machine Learning Model**
- Which model to choose
- Which hyperparameters to choose

**Training Process**
- Which loss function to choose
- Additional training parameters (model dependent)

**Evaluation**
- Which evaluation metrics to use
- How to measure success

---

## Classification Models

Where regression models fit a *line through* the data, classification models draw **borders around classes**. A new data point is classified by which side of the border it falls on.

Regression and classification are not completely distinct: you can encode classes as numbers (0 and 1 for a binary problem), learn a regression model, and round its output back to a class. Some models let you read that number as the **probability** of the class being correct.

### Logistic Regression
https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html

![[logistic-regression.png]]

A linear model squashed through the sigmoid so the output lands between 0 and 1:

$$
P(y = 1 \mid x) = \sigma(z) = \frac{1}{1 + e^{-z}}
\qquad\text{where}\qquad
z = w_1x_1 + \dots + w_nx_n + b
$$

Predict class 1 when $\sigma(z) \ge 0.5$, class 0 otherwise.

Prediction is **constant computation complexity** — it is just one weighted sum, no matter how big the training set was.

### K Nearest Neighbors
https://towardsdatascience.com/comprehensive-guide-to-approximate-nearest-neighbors-algorithms-8b94f057d6b6/?gi=a7db5c5fb29c

![[knn.png]]

Get the $k$ nearest neighbors from a point and try to guess where it belongs based on the neighbors and the distance from them.

$$
\hat{y} = \text{majority vote of } \{\, y_i : x_i \in N_k(x) \,\}
\qquad
d(x, x_i) = \sqrt{\sum_{j=1}^{n}(x_j - x_{ij})^2}
$$

There is no real "training" — the model *is* the dataset. It is more computationally expensive than logistic regression, because it needs to iterate through all points given. **Linear computation complexity** at prediction time.

### Support Vector Machine
https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47/

![[svm.png]]

Creates a buffer zone in between the groups and cuts a line through it in a way that the distance from the buffer zone edges is the same on all sides. It can be done in more than 2 dimensions.

$$
f(x) = \operatorname{sign}(w \cdot x + b)
\qquad
\text{maximise the margin } \frac{2}{\lVert w \rVert}
$$

Only the points sitting *on* the edge of the buffer zone — the **support vectors** — decide where the boundary goes. Every other point could move without changing the model.

### Decision Tree
https://towardsdatascience.com/understanding-random-forest-58381e0602d2/

![[decision-tree.png]]

A chain of yes/no questions on single features. Every split is axis-aligned, which makes the model very easy to read.

Prone to over-fit, because it recursively subdivides the dataset. If left unchecked, it can subdivide enough so that all data points are "correctly" classified — one leaf per data point. That is why we limit it with hyperparameters like `max_depth` or `min_samples_leaf`.

### Random Forest
https://towardsdatascience.com/understanding-random-forest-58381e0602d2/

![[random-forest.png]]

Put multiple models together (not necessarily the same) and let them all vote after they come to a conclusion.

Each tree sees a different random sample of the data and a different random subset of the features, so each one overfits in a *different* direction — and the mistakes cancel out in the vote. This is why a forest fixes the main weakness of a single decision tree.

---

## Confusion Matrix

It compares the true positives and true negatives with the false positives and false negatives.

![[confusion-matrix.png]]

It can be built for more than 2 options — an $n \times n$ matrix for $n$ classes. The main diagonal is the one we are more concerned about (true positives and true negatives).

$$
\text{accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

$$
\text{precision} = \frac{TP}{TP + FP}
\qquad\qquad
\text{recall} = \frac{TP}{TP + FN}
$$

- **Precision** — of everything I *flagged*, how much was right? (punishes false alarms)
- **Recall** — of everything that *was* there, how much did I catch? (punishes misses)

The **F1 score** is what happens when we are interested in more than one score. Say we are interested in recall and precision — it is the **harmonic** mean of the two:

$$
F_1 = 2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}
$$

The harmonic mean is used instead of the plain average $\frac{1}{2}(\text{precision} + \text{recall})$ because it stays low if *either* score is low — you cannot fake a good F1 by maxing out one and ignoring the other.

There are many other types of scores (table): https://en.wikipedia.org/wiki/F-score

---

## Neural Networks

A brain is a network of neurons that gets triggered based on the neurons that trigger before. Neural networks are a regression model inspired by how a brain works — part of a broader trend of copying concepts from biology into technology.

It does not really work like a brain at all.

### A Single Neuron

It consists of neurons. Each neuron takes inputs, multiplies them by weights, adds a bias, passes the result through a function, and produces an output:

![[neuron.png]]

$$
o = f(w_1 i_1 + w_2 i_2 + \dots + w_n i_n + b)
$$

- $i_x$ → input signals
- $w_x$ → **weights**: how much influence input $i_x$ has on the output
- $b$ → **bias**: how much signal the neuron needs before it is influenced
- $f$ → **activation function**: how the input is processed into the output

(it is exactly the way we calculate logistic regression — a function applied to a weighted sum of inputs)

A neuron is just a graphically nice way to express a regression model.

### The Activation Function

![[activation-functions.png]]

Any function is possible, but the choice matters:

| Function | Formula | Range |
|---|---|---|
| sigmoid | $\sigma(x) = \dfrac{1}{1 + e^{-x}}$ | $(0, 1)$ |
| tanh | $\tanh(x) = \dfrac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$ | $(-1, 1)$ |
| relu | $\max(0, x)$ | $[0, \infty)$ |
| linear | $x$ | $(-\infty, \infty)$ |

What plays a role when choosing:

- **Min and max of the function** — sometimes we want values in $(0,1)$, sometimes $(-1,1)$, sometimes unbounded.
- **Positive vs. negative opinions** — a strictly positive function only gathers *supporting* evidence for an outcome; one that can go negative can also find patterns that speak *against* it.
- **Calculation effort** — gradient descent needs the first derivative, so a function that is cheap to differentiate (like relu) trains much faster. This is the "computation trick" behind deep learning: relu is not differentiable at 0, but it is so cheap that we just pick a value there and move on.

> You usually pick a **monotonic** function (it steadily grows), and you **need some non-linearity**. If every function in the network is linear, the whole network collapses into a single linear function — and then you might as well have used linear regression in the first place.

### The Network

A neural network is a sequence of neurons connected to all others, arranged in layers:

![[neural-network-layers.png]]

- **Input layer** — calculates nothing, just provides the input part of $f_{learned}$.
- **Hidden layers** — intermediate calculation steps that process input into output.
- **Output layer** — provides the output part of $f_{learned}$.

In a **fully connected** layer — the go-to standard — all neurons in one layer are connected to all neurons in the next.

**Deep learning** is just a neural network with **more than one hidden layer**. It is largely a trend word for a slightly larger neural network.

Any 2-dimensional function can be approximated by a neural network (and very likely any higher-dimensional one too). Nice visual proof: http://neuralnetworksanddeeplearning.com/chap4.html

### Cost of That Power

They require a lot of learnable parameters and need high amounts of data and processing power. The network has to learn **one weight per edge and one bias per neuron**:

$$
\#\text{params} = \sum_{l} \left( n_{l-1} \cdot n_{l} + n_{l} \right)
$$

For the small $4 \rightarrow 3 \rightarrow 3 \rightarrow 2$ network in the diagram above:
$(4 \cdot 3 + 3) + (3 \cdot 3 + 3) + (3 \cdot 2 + 2) = 15 + 12 + 8 = \mathbf{35}$ parameters — for four inputs and two outputs.

That is the trade-off: very complex functions, but **prone to overfitting** and hungry for data. And the hyperparameters (layers, neurons per layer, activation functions, …) are all on you.

### The Neural Network Zoo

There are many types of neural net architectures. Saying *"I'm using a neural network"* says about as much as *"I'm using programming"*.

![[neural-network-zoo.png]]

Source: https://www.asimovinstitute.org/neural-network-zoo/

---

### Convolution Neural Networks (CNN)

Convolution in computer vision is a **filter**. Each pixel of an image is associated with a value, and if the matrix pattern has been detected, we can tell whether the image is something we are looking for or not.

The idea comes from how *we* detect faces: a face is two eyes, a nose and a mouth **in the right spatial arrangement**. An eye is in turn made of smaller circles and curves. And those are made of edges — which is exactly what a small filter matrix detects.

![[cnn-feature-hierarchy.png]]

A 3×3 edge-detection filter, slid over a greyscale image, produces a new value per pixel saying "how much diagonal line is here":

$$
\begin{bmatrix} -1 & -1 & -1 \\ -1 & -1 & 1 \\ -1 & 1 & 1 \end{bmatrix}
$$

**Why not just use a vanilla network?** You could — one input neuron per colour channel per pixel, one output neuron for face/no-face. Theoretically any function can be approximated that way. In practice it needs an enormous network and enormous amounts of training data. So instead we **restrict the architecture** to work the way our hand-made pipeline worked.

Two new layer types do that:

- **Convolution layers** slide a pattern over the image and produce a value for that pattern at each pixel.
- **Pooling layers** down-sample the network, reducing the resolution so that small differences in proportion, rotation and position even out.

![[pooling.png]]

Stacking them extracts increasingly higher-level features — edges → eyes/nose/mouth → face — and the result is fed into a normal neural network with a few hidden layers that produces the output:

![[cnn-example.png]]

![[cnn-diagram.png]]

In a CNN the pattern is detected in a 3D vector, RGB, where each layer detects one colour. The first 2 dimensions are the differences between the pixels and the 3rd dimension is the difference between the colours.

**CNNs are more efficient than normal neural networks**, because of two assumptions:

1. The **same pattern is applied to all pixels** of a layer. A 3×3 pattern is only 9 weights + 1 bias, no matter how large the image — a huge reduction compared to a fully connected layer.
2. **Pooling layers are not fully connected** either. Pooling four neurons needs only those four connections, and they don't even need weights, since the function is usually just taking the maximum.

Patterns are **not** encoded as matrices in the network. They are encoded by configuring the input weights and bias of a neuron — i.e. they work just the way neural networks work anyway.

Image processing still needs quite a lot of processing power and samples.

