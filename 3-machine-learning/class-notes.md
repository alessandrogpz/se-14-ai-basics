## Learning in Machine Learning Is Statistical

Machine learning (in a general sense) is deriving a function from examples. The more examples of the results we have, the better we can approximate a function that covers them.

When we have information about both the input and the output, it is called supervised learning.

### Types of Learning

- **Supervised Learning**: learns from samples of input and output.
- **Unsupervised Learning**: learns from samples of input only.
- **Reinforcement Learning**: no direct output is available, only a reinforcement signal.
  - e.g., moves in chess, which can only be judged good or bad after the whole game is finished.

---

## Supervised Learning

**Which type of function do we choose?**

- After choosing the type of function, we need to find the right parameters so that it matches our data points.

Neural networks are so popular because they can represent almost any function, no matter how complex.

When not using neural networks, the model is usually chosen by visualizing the data.

### The Loss Function

The sum of the absolute differences between each data point and the function line (measured vertically, at the same $x$), divided by the number of points:

$$
L = \frac{1}{n} \sum_{i=1}^{n} \left| y_i - f(x_i) \right|
$$

- $n$ → number of data points
- $x_i$ → input of the $i$-th data point
- $y_i$ → actual output of the $i$-th data point
- $f(x_i)$ → value predicted by the function (the point on the function line)
- $\left| y_i - f(x_i) \right|$ → absolute error of a single point

### Data Quality

Data science and machine learning go hand in hand: before even starting the machine learning process, we need really good quality data. Otherwise, it's "Garbage In → Garbage Out".

### Overfitting

When trying to fit every data point precisely, we risk overfitting the model. This means it becomes extremely specialized in that dataset and does not generalize well to new data.

To detect it, set aside part of the data only for testing, e.g., train on 80% of the data and test on the remaining 20%.

---

## Machine Learning - Choices

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
