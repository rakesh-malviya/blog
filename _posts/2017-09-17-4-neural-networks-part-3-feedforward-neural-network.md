---
title: '4. Neural Networks Part 3: Feedforward neural network'
layout: post
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
4. first hidden layer's input is $$x$$ such that $$x$$ is a vector $$(x_1,x_2,x_3,..... x_n)$$ of size $$n$$. **Note:** Some people consider an imaginary input layer of size $$n$$ which connects to first hidden layer. If you haven't read this anywhere just forget this note. For others I find this way of representing layers easier when will be implementing classes for different types of neural network layers.
5. We can have zero to any number of hidden layers. In case of zero hidden layers our network for binary classification task will be same as logistic regression descrsibed in previous post. Think ... &#x1f4ad; Since we have only single artificial neuron of output layer in the whole neural network.

![]({{ site.baseurl }}/assets/img/blog/4/4_1_neuralnetwork.svg "Feedforward neural network")
<i><center>Fig 1 : Feedforward neural network</center></i>

#### Establish Notations:
1. For notational convenience we have removed superscript (i) for $$i^{th}$$ training example $$(x,y)$$. So $$x$$ is single training input, $$y$$ is expected output and $$y'$$ is predicted output
2. Input $$x$$ is such that $$x$$ is a vector $$(x_1,x_2,x_3,..... x_n)$$ of size $$n$$.
3. We have $$l$$ hidden layers. Output layer is denoted by $$l+1$$ or $$o$$
4. Each layer $$l$$ has following attributes weight matrix $$W^l$$, output vector $$h^l$$ and bias vector $$b^l$.
5. **I am proposing below notations for your easy of understanding the equations in upcoming sections. Also check Fig 1 above**
6. let output layer has size $$r$$ i.e $$r$$ artificial neurons. Layer $$l$$ has size $$q$$ and Layer $$l-1$$ has size $$p$$
7. We will use $$i$$ to denote individual neurons in layer $$l-1$$, $$j$$ for layer $$l$$ and $$k$$ for output layer $$o$$.
8. **You need to think carefully why I am proposing below sizes for weight matrix, output vector and bias. It will help if you assume this, read understand whole article, comeback to this and think**.
9. For layer  $$l$$ weight matrix $$W^l$$ has size $$[qxp]$$ and for ouput layer $$W^o$$ has size $$[rxq]$$
10.  For layer  $$l-1$$ vectors $$h^{l-1}$$ and $$b^{l-1}$$ has size $$p$$ and  for layer  $$l$$ vectors $$h^l$$ and $$b^l$$ has size $$q$$ and for output layer $$o$$ vectors $$h^o$$ and $$b^o$$ has size $$r$$
11.  $$j^{th}$$ Artificial neuron in layer $$l$$  has bias $$b_j^l$$ (scalar), output $$h_j^l$$(scalar),  weight vector $$W_{j:}^l$$ , i.e. $$j^{th}$$ row of weight matrix $$W^l$$ for layer l. *It will help it you remember from my previous post on logistic reggression that each neuron has weight vector, (scalar or single value) bias and output*.