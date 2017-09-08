---
title: '2. Neural Networks Part 1: Logistic Regression (Least Square Error)'
layout: post
tags:
- Neural Networks
- Machine Learning
- Deep Learning
---

Required Learning: Linear regression basics [link](http://www.holehouse.org/mlclass/01_02_Introduction_regression_analysis_and_gr.html) 

We are starting from basic unit of Neural networks single activation neuron. A Neural network with single neuron is same as logistic regression. Therefore a neural network can be considered as a networked set of logistic regression units.

### Establish notations for future use<sup>[1](#references)</sup>
1. $$x^{(i)}$$ to denote the i<sup>th</sup> “input ” of training data
2. $$y^{(i)}$$ to denote the i<sup>th</sup> “output” or target of training data
3. Pair $$(x^{(i)}, y^{(i)})$$ is called a training example
4. The dataset that we’ll be using to learn—a list of m training examples $$\{(x(i), y(i)); i = 1, . . . , m\}$$ — is called a training set
5. Each $$x^{(i)}$$ in training set can have $$n$$ **features** such that $$x^{(i)}$$ is a vector $$(x^{(i)}_1,x^{(i)}_2,x^{(i)}_3,..... x^{(i)}_n)$$
6. In current setup of logistic regression $$y^{(i)}$$ is scalar value


**Note that the superscript “(i)” in the notation is simply an index into the training set, and has nothing to do with exponentiation.**


![]({{ site.baseurl }}/assets/img/blog/2/2_1_logistic_reg.svg "Single neuron")
<i><center>Fig: Single neuron</center></i>


Let $$y'^{(i)} = f(u^{(i)})$$ , 
where 
1. $$y'^{(i)}$$ is predicted output
2. $$f$$ is activation function
3. $$u^{(i)} = {b} + \sum_{j=1}^{n} {w_j}\cdot{x_j^{(i)}}$$, for $$i^{th}$$ training example, where $$b$$ is bias of the neuron.
4. $$w_j$$ **weights** or **training parameters** we need to learn

We will apply gradient descent to minimize the squared error cost function $$J$$, also called least square error

\begin{align}
J = \sum_{i=1}^{m} {\dfrac{1}{2}}(y^{(i)} - y'^{(i)})^2 \tag{1} \label{eq1}
\end{align}


We will use Sigmoid function as activation function, i.e. $$\sigma(u)$$
\begin{align}
\sigma(u^{(i)}) = \dfrac{1}{1+e^{-u^{(i)}}} = f(u^{(i)}) = y'^{(i)}   \tag{2} \label{eq2}
\end{align}

#### Derivatives

\begin{align}
\frac{\partial\sigma(u)}{\partial{u}} = \sigma(u)\cdot(1 - \sigma(u))   \tag{3} \label{eq3}
\end{align}
\begin{align}
\frac{\partial{J}}{\partial{y'^{(i)}}} = y'^{(i)} - y^{(i)}   \tag{4} \label{eq4}
\end{align}
\begin{align}
\frac{\partial{u^{(i)}}}{\partial{w_j}} = x_j^{(i)}   \tag{5} \label{eq5}
\end{align}
\begin{align}
\frac{\partial{u^{(i)}}}{\partial{b}} = 1   \tag{6} \label{eq6}
\end{align}

#### Gradient Descent

To learn $$w_j$$ and $$b$$ parameter so that $$J$$ is minimum. We will find $$\frac{\partial{J}}{\partial{w_j}}$$ and $$\frac{\partial{J}}{\partial{b}}$$

**Note: $$J$$ is our loss function and $$j$$ is for indexing**

Since, $$J$$ is function of $$y'^{(i)}$$, $$y'^{(i)}$$ is function of $$u^{(i)}$$ and $$u^{(i)}$$ is function of $$w_j$$ by $$\eqref{eq1} and \eqref{eq2}$$. Using chain rule of differentiation

$$
\begin{align} 
\frac{\partial{J}}{\partial{w_j}} &= \sum_{i=1}^{m}  \frac{\partial{J}}{\partial{y'^{(i)}}} \cdot  \frac{\partial{y'^{(i)}}}{\partial{w_j}}   && \text{by \eqref{eq1}} \\
&= \sum_{i=1}^{m}  \frac{\partial{J}}{\partial{y'^{(i)}}} \cdot  \frac{\partial{y'^{(i)}}}{\partial{u^{(i)}}} \cdot    \frac{\partial{u^{(i)}}}{\partial{w_j}}  \\
&= \sum_{i=1}^{m}  (y'^{(i)} - y^{(i)}) \cdot  \frac{\partial{y'^{(i)}}}{\partial{u^{(i)}}} \cdot    \frac{\partial{u^{(i)}}}{\partial{w_j}}  && \text{by \eqref{eq4}} \\
&= \sum_{i=1}^{m}  (y'^{(i)} - y^{(i)}) \cdot  y'^{(i)} \cdot (1 - y'^{(i)}) \cdot    \frac{\partial{u^{(i)}}}{\partial{w_j}}  && \text{by \eqref{eq2} and \eqref{eq3}} \\
\frac{\partial{J}}{\partial{w_j}} &= \sum_{i=1}^{m}  (y'^{(i)} - y^{(i)}) \cdot  y'^{(i)} \cdot (1 - y'^{(i)}) \cdot  x_j^{(i)} && \text{by \eqref{eq5}} \tag{7} \label{eq7}
\end{align}
$$

So during the training update equation for $$w_j$$ over all $$j = 1 ..... n$$ is as follows:

$$
\begin{align} 
w_j &= w_j - \eta \cdot \frac{\partial{J}}{\partial{w_j}} \\
&= w_j - \eta \cdot \sum_{i=1}^{m}  (y'^{(i)} - y^{(i)}) \cdot  y'^{(i)} \cdot (1 - y'^{(i)}) \cdot  x_j^{(i)} && \text{by \eqref{eq7}}
\end{align}
$$

Similarly:

$$
\begin{align} 
\frac{\partial{J}}{\partial{b}} &= \sum_{i=1}^{m}  \frac{\partial{J}}{\partial{y'^{(i)}}} \cdot  \frac{\partial{y'^{(i)}}}{\partial{b}}   && \text{by \eqref{eq1}} \\
&= \sum_{i=1}^{m}  \frac{\partial{J}}{\partial{y'^{(i)}}} \cdot  \frac{\partial{y'^{(i)}}}{\partial{u^{(i)}}} \cdot    \frac{\partial{u^{(i)}}}{\partial{b}}  \\
&= \sum_{i=1}^{m}  (y'^{(i)} - y^{(i)}) \cdot  \frac{\partial{y'^{(i)}}}{\partial{u^{(i)}}} \cdot    \frac{\partial{u^{(i)}}}{\partial{b}}  && \text{by \eqref{eq4}} \\
&= \sum_{i=1}^{m}  (y'^{(i)} - y^{(i)}) \cdot  y'^{(i)} \cdot (1 - y'^{(i)}) \cdot    \frac{\partial{u^{(i)}}}{\partial{b}}  && \text{by \eqref{eq2} and \eqref{eq3}} \\
\frac{\partial{J}}{\partial{b}} &= \sum_{i=1}^{m}  (y'^{(i)} - y^{(i)}) \cdot  y'^{(i)} \cdot (1 - y'^{(i)}) && \text{by \eqref{eq6}} \tag{8} \label{eq8}
\end{align}
$$

Update equation for bias $$b$$ is as follows:

$$
\begin{align} 
b &= b - \eta \cdot \frac{\partial{J}}{\partial{w_j}} \\
&= b - \eta \cdot \sum_{i=1}^{m}  (y'^{(i)} - y^{(i)}) \cdot  y'^{(i)} \cdot (1 - y'^{(i)}) && \text{by \eqref{eq8}}
\end{align}
$$



Training Steps:
1. Choose an initial vector of parameters  $$w = (w_1,......w_n)$$, bias $$b$$ and learning rate $$\eta$$.
2. Repeat for predifined epoch such that approximate minimum $$J$$ loss is obtained:
	1. Evaluate and store $$y'^{(i)}$$ for all $$i = 1,2,3...m$$ training examples by using equation $$\eqref{eq2}$$
	2. Update bias, $$b = b - \eta \cdot \sum_{i=1}^{m}  (y'^{(i)} - y^{(i)}) \cdot  y'^{(i)} \cdot (1 - y'^{(i)})$$			
	3. 	For $$ j = 1,2,.....n $$ in $$w$$ : 
			1. Update, $$w_j = w_j - \eta \cdot \sum_{i=1}^{m}  (y'^{(i)} - y^{(i)}) \cdot  y'^{(i)} \cdot (1 - y'^{(i)}) \cdot  x_j^{(i)} $$

[here](https://github.com/rakesh-malviya/blog/blob/master/code/notebooks/3-neural-networks-part-1-logistic-regression-least-square-error.ipynb) is the python implementation of this article.

## References:
1. http://cs229.stanford.edu/notes

## Previous Posts:
["1. Word2vec Part 1: Basics"]({{ site.baseurl }}{% post_url  2017-08-15-word2vec-basics %})