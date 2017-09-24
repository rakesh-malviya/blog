---
title: '4. Neural Networks Part 3: Feedforward neural network'
layout: post
comments: true
tags:
- Neural Networks
- Machine Learning
- Deep Learning
---

**Required Learning:** My previous posts on Neural networks

**Feedforward neural network** is neural network architecture consisting of layers of artificial neurons such that:
1. Each neuron in a layer is connected to all neurons of previous layer, i.e. its input is output of all neurons of previous layers.
2. Last layer is called output layer and returns $$y'$$, predicted y
3. Other layers are called hidden layers
4. first hidden layer's input is $$\mathbf{x}$$ such that $$\mathbf{x}$$ is a vector $$(x_1,x_2,x_3,..... x_n)$$ of size $$n$$. **Note:** Some people consider an imaginary input layer of size $$n$$ which connects to first hidden layer. If you haven't read this anywhere just forget this note. For others I find this way of representing layers easier when we will be implementing classes for different types of neural network layers.
5. We can have zero to any number of hidden layers. In case of zero hidden layers our network for binary classification task will be same as logistic regression descrsibed in previous post. Think ... &#x1f4ad; Since we have only single artificial neuron of output layer in the whole neural network.

![]({{ site.baseurl }}/assets/img/blog/4/4_1_neuralnetwork.svg "Feedforward neural network")
<i><center>Fig 1 : Feedforward neural network</center></i>

#### Establish Notations:
1. For notational convenience we have removed superscript (i) for $$i^{th}$$ training example $$(x,y)$$. So $$x$$ is single training input, $$y$$ is expected output and $$y'$$ is predicted output
2. Input $$\mathbf{x}$$ is such that $$\mathbf{}x$$ is a vector $$(x_1,x_2,x_3,..... x_n)$$ of size $$n$$.
3. We have $$l$$ hidden layers. Output layer is denoted by $$l+1$$ or $$o$$
4. Each layer $$l$$ has following attributes weight matrix $$\mathbf{W}^l$$, output vector $$\mathbf{h}^l$$ and bias vector $$\mathbf{b}^l$.
5. **I am proposing below notations for your easy of understanding the equations in upcoming sections. Also check Fig 1 above**
6. let output layer has size $$r$$ i.e $$r$$ artificial neurons. Layer $$l$$ has size $$q$$ and Layer $$l-1$$ has size $$p$$
7. We will use $$i$$ to denote individual neurons in layer $$l-1$$, $$j$$ for layer $$l$$ and $$k$$ for output layer $$o$$.
8. **You need to think carefully why I am proposing below sizes for weight matrix, output vector and bias. It will help if you assume this, read understand whole article, comeback to this and think**.
9. For layer  $$l$$ weight matrix $$\mathbf{W}^l$$ has size $$[qxp]$$ and for ouput layer $$\mathbf{W}^o$$ has size $$[rxq]$$
10.  For layer  $$l-1$$ vectors $$\mathbf{h}^{l-1}$$ and $$\mathbf{b}^{l-1}$$ has size $$p$$ and  for layer  $$l$$ vectors $$\mathbf{h}^l$$ and $$\mathbf{b}^l$$ has size $$q$$ and for output layer $$o$$ vectors $$\mathbf{h}^o$$ and $$\mathbf{b}^o$$ has size $$r$$
11.  $$j^{th}$$ Artificial neuron in layer $$l$$  has bias $$b_j^l$$ (scalar), output $$h_j^l$$(scalar),  weight vector $$\mathbf{W}_{j:}^l$$ , i.e. $$j^{th}$$ row of weight matrix $$\mathbf{W}^l$$ for layer l. Also weight vector $$\mathbf{W}_{j:}^l = (w_{j1},w_{j2},....w_{ji}...,w_{jp},)$$. *It will help it you remember from my previous post on logistic reggression that each neuron has weight vector, (scalar or single value) bias and output*.
12.  **Below notations are neccessary to easy of understand and implementing mathematical equations in upcoming sections**
13.  We will use $$\odot$$ for elementwise multiplication of two vectors, for example $$\left[\begin{array}{c} 1 \\ 2 \end{array}\right]   \odot \left[\begin{array}{c} 1 \\ 2\end{array} \right]= \left[ \begin{array}{c} {1 \cdot 1} \\ {2 \cdot 2} \end{array} \right]= \left[ \begin{array}{c} 1 \\ 4 \end{array} \right]$$
14.  We will use $$\otimes$$ for matrix multiplication
15.  We will use bold capital letters for matrix e.g. $$\mathbf{W}^l$$, bold small letter for vectors $$\mathbf{h}^l$$ and normal small latters for scalars.

#### Training
Let us use our Neural network (in Fig 1) for classification. Note, for binary classification our last layer will have only one neuron hence $$r==1$$, but we will keep our approach general. Also for simplicity we will assume that activation function of all the neurons is sigmoid function $$\sigma()$$.

Let our loss function, $$J = \frac{1}{2}\sum_{k=1}^{r}(y_k - y'_k)^2$$.

So our goal is to minimize loss, $$J$$. We can do this only by learning correct weights and bias for each neuron in our network. It looks complex if we look at each neuron individualy. We can simplify this task by breaking our training steps into three steps of **Backpropagation**:
1. **Forward pass:** calculate output of each neuron.
2. **Backward pass:** calculate $$\frac{\partial{J}}{\partial{b_j^l}}$$ and $$\frac{\partial{J}}{\partial{w_{ji}^l}}$$ for each neuron.
3. **Update weights and biases**

#### Foward pass:
Output of $$j^{th}$$ neuron in $$l^{th}$$ layer is defined as, 
$$
\begin{align} 
h_j^l = \sigma(\sum_{i=1}^{p} w_{ji}^l  h_i^{l-1} + b_j^l) \tag{1} \label{eq1}
\end{align}
$$

**We can use equation $$\eqref{eq1}$$ to get output of each neuron in forward pass**

![]({{ site.baseurl }}/assets/img/blog/4/4_2_forward_pass.svg "Forward pass")
<i><center>Fig 2 : Forward pass </center></i>

Let total input to above neuron defined as, 
$$
\begin{align} 
z_j^l = \sum_{i=1}^{p} w_{ji}^l  h_i^{l-1} + b_j^l \tag{2} \label{eq2}
\end{align}
$$

hence, 
$$
\begin{align} 
h_j^l &= \sigma(z_j^l) && \text{by \eqref{eq1} and \eqref{eq2}} \tag{3} \label{eq3}
\end{align}
$$

similarly for output layer, 
$$
\begin{align} 
h_k^o &= \sigma(z_k^o) && \text{by \eqref{eq3}} \tag{4} \label{eq4}\\
z_k^o &= \sum_{j=1}^{q} w_{kj}^o  h_j^{l} + b_k^o && \text{by \eqref{eq2}} \tag{5} \label{eq5}
\end{align}
$$

Also, output of output layer is predicted $$y$$ hence,

$$
\begin{align} 
h_k^o = y'_k \tag{6} \label{eq6}
\end{align}
$$

###### Forward pass (vectorized)
It is important that we implement code in the vectors and matrix operations to improve preformance.

$$
\begin{align} 
\mathbf{h}^l &= \sigma(\mathbf{z}^l) && \text{by \eqref{eq3}} \tag{7} \label{eq7}\\
\mathbf{z}^l &= \mathbf{W}^l \otimes \mathbf{h}^{l-1} + \mathbf{b}^l && \text{by \eqref{eq2}}  \tag{8} \label{eq8}\\
\mathbf{h}^o &= \sigma(\mathbf{z}^o) && \text{by \eqref{eq4}} \tag{9} \label{eq9}\\
\mathbf{z}^o &= \mathbf{W}^o \otimes \mathbf{h}^{l} + \mathbf{b}^o && \text{by \eqref{eq5}}  \tag{10} \label{eq10}\\
\end{align}
$$

**Note: for vectors $$\sigma()$$ is called elementwise**

###### Useful derivatives:
$$
\begin{align} 
\frac{\partial{h_j^l}}{\partial z_j^l} &= \frac{\partial}{\partial z_j^l}{\sigma({z_j^l}}) && \text{by \eqref{eq3}} \\
\frac{\partial{h_j^l}}{\partial z_j^l} &= \sigma'({z_j^l}) && \tag{11} \label{eq11} \\
\frac{\partial{z_j^l}}{\partial w_{ji}^l} &= h_i^{l-1} && \tag{12} \label{eq12}
\end{align}
$$

where, $$\sigma'(z)$$ is derivative of $$\sigma(z)$$ with repect $$z$$

Similarly for loss $$J$$,
$$
\begin{align} 
\frac{\partial{J}}{\partial y'_k} &= (y'_k - y_k) && \text{how ? check prev post} \tag{13} \label{eq13}
\end{align}
$$




#### Backward pass

 Gradient of loss with respect to weights for output layer is,  
$$
\begin{align} 
\frac{\partial{J}}{\partial{w_{kj}^o}} &= \frac{\partial{J}}{\partial{y'_k}} \cdot \frac{\partial{y'_k}}{\partial{z_k^o}} \cdot \frac{\partial{z_k^o}}{\partial{w_{kj}^o}}  &&\text{by chain rule} \tag{14} \label{eq14} \\
\frac{\partial{J}}{\partial{w_{kj}^o}} &=(y'_k - y_k)  \cdot  \sigma'({z_k^o}) \cdot h_j^l  &&\text{by \eqref{eq11},\eqref{eq12},\eqref{eq13}}  \tag{15} \label{eq15}\\
\end{align} 
$$

Let us define gradient of loss with respect to total input for output layer as,

$$
\begin{align} 
\delta_k^o &= \frac{\partial{J}}{\partial{z_k^o}} \tag{16} \label{eq16}\\
&= \frac{\partial{J}}{\partial{y'_k}} \cdot \frac{\partial{y'_k}}{\partial{z_k^o}} \\
\delta_k^o &= \frac{\partial{J}}{\partial{y'_k}} \cdot \sigma'({z_k^o}) \tag{17} \label{eq17}\\
\frac{\partial{J}}{\partial{w_{kj}^o}} &=\delta_k^o \cdot h_j^l  &&\text{by \eqref{eq15},\eqref{eq17}} \tag{18} \label{eq18}\\
\end{align} 
$$

Similarly for layer $$l$$,

$$
\begin{align} 
\frac{\partial{J}}{\partial{w_{ji}^l}} &=\delta_j^l \cdot h_i^{l-1} \tag{19} \label{eq19} \\
\delta_j^l &= \frac{\partial{J}}{\partial{z_j^l}} \\
&= \sum_{k=1}^{r} \frac{\partial{J}}{\partial{z_k^o}} \cdot  \frac{\partial{z_k^o}}{\partial{z_j^l}}\\
&= \sum_{k=1}^{r} \frac{\partial{J}}{\partial{z_k^o}} \cdot  \frac{\partial}{\partial{z_j^l}} \big( \sum_{j=1}^{q} w_{kj}^o  h_j^{l} + b_k^o \big) && \text{by \eqref{eq5}}\\
&= \sum_{k=1}^{r} \frac{\partial{J}}{\partial{z_k^o}} \cdot w_{kj}^o \cdot  \frac{\partial{h_j^l}}{\partial{z_j^l}} \\
\delta_j^l &= \sum_{k=1}^{r} \delta_k^o \cdot w_{kj}^o \cdot \sigma'({z_j^l}) && \text{by \eqref{eq11},\eqref{eq16}}\\
\delta_j^l &= \sum_{k=1}^{r} \delta_k^{l+1} \cdot w_{kj}^{l+1} \cdot \sigma'({z_j^l}) && \text{by,  } {o=l+1} \tag{20} \label{eq20}\\
\end{align} 
$$
	
Similarly you can easily prove for bias,
$$
\begin{align} 
\frac{\partial{J}}{\partial{b_j^l}} &=\delta_j^l \tag{21} \label{eq21}\\
\frac{\partial{J}}{\partial{b_k^o}} &=\delta_k^o \tag{22} \label{eq22}\\
\end{align} 
$$


###### Backword pass (Vectorized)

$$
\begin{align} 
\pmb{\delta}^o &= \nabla_{\mathbf{y'}}J \odot \sigma'(\mathbf{z}^o) && \text{by \eqref{eq17}} \tag{23} \label{eq23}\\
\pmb{\delta}^l &= ((\mathbf{W}^{l+1})^T \otimes \pmb{\delta}^{l+1} )\odot  \sigma'({\mathbf{z}^l}) && \text{by \eqref{eq20}} \tag{24} \label{eq24}\\
\end{align} 
$$

where $$(\mathbf{W}^l)^T$$ is transpose of matrix $$\mathbf{W}^l$$  and $$\nabla_{\mathbf{y'}}J$$, Derivative of J with respect recpect to vector $$\mathbf{y'}$$, i.e.

$$
\begin{align} 
\nabla_{\mathbf{y'}}J =  \frac{\partial{J}}{\partial{\mathbf{y'}}}= \left[\begin{array}{c} \frac{\partial{J}}{y'_1} \\  \frac{\partial{J}}{y'_2} \\ \vdots \\ \frac{\partial{J}}{y'_k} \\ \vdots \end{array}\right]
\end{align} 
$$

Also,

$$
\begin{align}
\nabla_{\mathbf{b}^l}J &=  \pmb{\delta}^l   && \text{by \eqref{eq21}} \tag{25} \label{eq25}\\
\nabla_{\mathbf{W}^l}J &=  \pmb{\delta}^l  \otimes (\mathbf{h}^{l-1})^T && \text{by \eqref{eq19}} \tag{26} \label{eq26}\\
\end{align} 
$$

Where, $$\nabla_{\mathbf{W}^l}J$$, Derivative of J with respect recpect to matrix $$\mathbf{W}^l$$, i.e.

$$
\begin{align}
\nabla_{\mathbf{W}^l}J =  \frac{\partial{J}}{\partial{\mathbf{W}^l}}  = \begin{bmatrix} \frac{\partial{J}}{\partial{w_{11}}} & \frac{\partial{J}}{\partial{w_{12}}} & \cdots & \frac{\partial{J}}{\partial{w_{1i}}} & \cdots \\ \frac{\partial{J}}{\partial{w_{21}}} & \frac{\partial{J}}{\partial{w_{22}}} & \cdots & \frac{\partial{J}}{\partial{w_{2i}}} & \cdots \\ \vdots & \vdots & \vdots & \vdots \\ \frac{\partial{J}}{\partial{w_{j1}}} & \frac{\partial{J}}{\partial{w_{j2}}} & \cdots & \frac{\partial{J}}{\partial{w_{ji}}} & \cdots \\ \vdots & \vdots & \vdots & \vdots \end{bmatrix}
\end{align} 
$$

#### Update step:
1. Given a training batch of size $$m$$ and learning rate $$\eta$$
2. For each training example in $$i$$ in batch do **Forward pass** and Backward pass, accumulate $$\nabla_{\mathbf{b}^l}J^{(i)}$$ and $$\nabla_{\mathbf{W}^l}J^{(i)}$$
3. Update weights and bias for each layer $$l$$ as follows, 

$$
\begin{align}
\mathbf{b}_l &= \mathbf{b}_l -\eta \cdot \frac{1}{m} \cdot \sum_{i=1}^{m} \nabla_{\mathbf{b}^l}J^{(i)} \tag{27} \label{eq27}\\
\mathbf{W}_l &= \mathbf{W}_l - \eta \cdot \frac{1}{m} \cdot \sum_{i=1}^{m} \nabla_{\mathbf{W}^l}J^{(i)} \tag{28} \label{eq28}\\
\end{align} 
$$

#### td;dr.

$$
\begin{align}
\mathbf{h}^l &= \sigma(\mathbf{z}^l) && \text{by \eqref{eq7}} \\
\mathbf{z}^l &= \mathbf{W}^l \otimes \mathbf{h}^{l-1} + \mathbf{b}^l && \text{by \eqref{eq8}}  \\
\pmb{\delta}^o &= \nabla_{\mathbf{y'}}J \odot \sigma'(\mathbf{z}^o) && \text{by \eqref{eq23}} \\
\pmb{\delta}^l &= ((\mathbf{W}^{l+1})^T \otimes \pmb{\delta}^{l+1} )\odot  \sigma'({\mathbf{z}^l}) && \text{by \eqref{eq24}} \\
\nabla_{\mathbf{b}^l}J &=  \pmb{\delta}^l   && \text{by \eqref{eq26}} \\
\nabla_{\mathbf{W}^l}J &=  \pmb{\delta}^l  \otimes (\mathbf{h}^{l-1})^T && \text{by \eqref{eq26}} \\
\mathbf{b}_l &= \mathbf{b}_l -\eta \cdot \frac{1}{m} \cdot \sum_{i=1}^{m} \nabla_{\mathbf{b}^l}J^{(i)} && \text{by \eqref{eq27}}  \\
\mathbf{W}_l &= \mathbf{W}_l - \eta \cdot \frac{1}{m} \cdot \sum_{i=1}^{m} \nabla_{\mathbf{W}^l}J^{(i)} && \text{by \eqref{eq28}} \\
\end{align} 
$$

#### [Python Notebook](https://github.com/rakesh-malviya/MLCodeGems/blob/master/notebooks/Neural_networks/4-neural-networks-part-3-feedforward-neural-network.ipynb)

#### [Code](https://github.com/rakesh-malviya/MLCodeGems/tree/master/Projects/Neural_networks/src)

[Here](https://github.com/rakesh-malviya/MLCodeGems/tree/master/Projects/Neural_networks/src) is the python implementation of the above article.

#### References:
1. Neural Networks and Deep Learning Chapter 2 [link](http://neuralnetworksanddeeplearning.com/chap2.html)