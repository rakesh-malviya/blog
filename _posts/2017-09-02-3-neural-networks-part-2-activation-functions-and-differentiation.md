---
title: '3. Neural Networks Part 2: Activation functions and differentiation'
layout: post
crosspost_to_medium: true
tags:
- Neural Networks
- Machine Learning
- Deep Learning
---

## Activation functions

A neural network is a network of artificial neurons connected to each other in a specific way. Job of neural network is to learn from given data. The prediction function that neural network must learn can be non-linear. Activation function in artificial neurons helps the neural network to learn non-linear prediction function.

Linear prediction             |  Non-linear prediction<sup>[1](#references)</sup>
:-------------------------:|:-------------------------:
![]({{site.baseurl}}/assets/img/blog/3/3_1_linear.png)  | ![]({{site.baseurl}}/assets/img/blog/3/3_2_nonlinear.png)

Activation functions (generally) have functional form of $$f(u) = f(w^{T}{x} + b)$$, where $$u = {b} + \sum_{j=1}^{n} {w_j}\cdot{x_j} = w^{T}{x} + b$$, $$w = (w_1,w_2,.......w_n)$$ weight vector  and $$x= (x_1,x_2,.....x_n)$$ single training data vector

#### 1. Sigmoid activation function

A sigmoid function, $$f(u) = \frac{1}{1+e^{-u}}$$.  It takes a real-valued number and “squeeze” it into range between 0 and 1. Large negative numbers become $$\approx 0$$ and large positive numbers become $$\approx 1$$. 

##### Pros: 
For classification problem it is used as activation of output layer of a neural network.

##### Cons:
1. **Saturate and kill gradients:** When neuron's activation saturates at 1 or 0 , the gradient becomes almost zero. Which will make the neuron unable to backpropagate and learn.
2. **Outputs are not zero-centered:** Since outputs are in range 0 to 1 neurons in next layer will receive data that is not zero centered. Hence gradient of weights $$w$$ during backpropagation will be either all positive or all negative, which can cause undesirable zig-zagging dynamics in gradient updates of weights. When considering gradient added over all training data in a batch this problem is not much severe compared to "Saturate and kill gradients"

#### 2. Tanh activation function
A tanh function, $$f(u) = \frac{e^{u}-e^{-u}}{e^{u}+e^{-u}} = \frac{sinh(u)}{cosh(u)}$$.  It takes a real-valued number and “squeeze” it into range between -1 and 1. Large negative numbers become $$\approx -1$$ and large positive numbers become $$\approx 1$$. 

##### Pros:
It is preferred over sigmoid because its outputs are zero centered


#### 3. ReLU activation function 
The Rectified Linear Unit, ReLU is $$f(u) = max(0,u)$$

##### Pros:
1. Greatly increase training speed compared to tanh and sigmoid
2. Less expensive computations compared to tanh and sigmoid
3. Reduces likelihood of the gradient to vanish. Since when $$u > 0 $$, the gradient has constant value.
4. **Sparsity:** When more $$u <= 0$$, the $$f(u)$$ can be more sparse

##### Cons:
1. Tend to blow up activation (there is no mechanism to constrain the output of the neuron, as $$u$$ itself is the output).
2. **Closed ReLU or Dead ReLU**: If inputs tend to make $$u<=0$$ than the most of the neurons will always have 0 gradient updates hence closed or dead.

#### 4. Leaky ReLU:

It solves the dead ReLU problem. $$0.01$$ is coefficient of leakage.  Leaky ReLU is as follows: 

$$
    f(u)= 
\begin{cases}
    x,& \text{if } x > 0\\
    (0.01)x,              & \text{otherwise}
\end{cases}
$$

#### 5. Parameterized ReLU or PReLU:
Parameterizes coefficient of leakage $$\alpha$$ in Leaky ReLU.
$$
    f(u)= 
\begin{cases}
    x,& \text{if } x > 0\\
    \alpha{x},              & \text{otherwise}
\end{cases}
$$

#### 6. Maxout
Generalization of ReLU, Leaky ReLU and PReLU. It does not have functional form of $$f(u) =  f(w^{T}{x} + b)$$, instead it computes function $$max({w'^T}{x} + b',{w^T}{x} + b)$$. 

##### Pros:
Maxout has pros of ReLU but doesn't have dead ReLU issue

##### Cons:
It has twice number of weight parameters to learn $$w'$$ and $$w$$

## What Activation function should I use ?<sup>[2](#references)</sup>
1. For output layer use sigmoid if classification task
2. For output layer use no activation or Purelin function $$f(u) = u$$ if regression task
3. For other neurons: 
    1. Use the ReLU non-linearity if you carefully set learning rates and monitor the fraction of “dead ReLU” in network.
    2. Else try Leaky ReLU or Maxout.
    3. Or try tanh but it will work worse than ReLU
    4. Never use sigmoid

## Differentiation:
#### Basic formulas:
Given $$f(x)$$ and $$g(x)$$ are differentiable functions (the derivative exists), $$c$$ and $$n$$ are any real numbers:
$$
\begin{align} 
\frac{d}{dx}f(x) &= f'(x) \tag{1} \label{eq1} \\
\frac{d}{dx}g(x) &= g'(x) \tag{2} \label{eq2} \\
\frac{d}{dx}(f(x) \pm g(x)) &= \frac{d}{dx}f(x) \pm \frac{d}{dx}g(x)  \\
 &=  f'(x) \pm g'(x) \tag{3}  \label{eq3} \\
\frac{d}{dx}x^n &= nx^{n-1}  && \text{power-rule} \tag{4} \label{eq4} \\
\frac{d}{dx} f(x)g(x) &= f'(x)g(x) + f(x)g'(x) && \text{product-rule} \tag{5} \label{eq5} \\ 
\frac{d}{dx} \Bigg [\frac{f(x)}{g(x)}\Bigg ]  &= \frac{f'(x)g(x)-g'(x)f(x)}{g^{2}(x)} && \text{Quotient Rule} \tag{6} \label{eq6} \\ 
\frac{d}{dx} f(g(x))  &= f'(g(x))g'(x) && \text{Chain Rule} \tag{7} \label{eq7} \\
\frac{d}{dx} c &= 0 \tag{8} \label{eq8} \\
tanh(x) &= \frac{sinh(x)}{cosh(x)}  \tag{9} \label{eq9} \\
\frac{d}{dx}sinh(x) &= cosh(x)  \tag{10} \label{eq10} \\
\frac{d}{dx}cosh(x) &= sinh(x)  \tag{11} \label{eq11} \\
\frac{d}{dx}e^x &= e^x  \tag{12} \label{eq12} \\
\end{align}
$$

#### Sigmoid function:
$$
\begin{align} 
\frac{d}{dx} f(x) &= \frac{d}{dx} \Bigg[ \frac{1}{1+e^{-x}} \Bigg] \\
&= \frac{d}{dx} \bigg[ \frac{e^x}{1+e^x} \bigg] \\ 
&= \frac {\bigg(\frac{d}{dx}e^x\bigg)(1+e^x) - \bigg(\frac{d}{dx}(1 + e^x)\bigg)(e^x)}{(1+e^x)^2} &&  \text{by \eqref{eq6}} \\
&= \frac {\bigg(\frac{d}{dx}e^x\bigg)(1+e^x) - \bigg(\frac{d}{dx} 1 +\frac{d}{dx} e^x\bigg)(e^x)}{(1+e^x)^2} && \text{by \eqref{eq3}} \\
&= \frac {(e^x)(1+e^x) - (e^x)(e^x)}{(1+e^x)^2} && \text{by \eqref{eq8} and \eqref{eq12}} \\
&= \bigg[\frac {(e^x)}{(1+e^x)}\bigg]  - \bigg[\frac{(e^x)^2}{(1+e^x)^2}\bigg] \\
&= \bigg[\frac {e^x}{1+e^x}\bigg]  - \bigg[\frac{e^x}{1+e^x}\bigg]^2 \\
&= \bigg[\frac {1}{1+e^{-x}}\bigg]  - \bigg[\frac{1}{1+e^{-x}}\bigg]^2 \\
& = f(x) - (f(x))^2 \\
& = f(x) (1 - f(x)) \\
\end{align}
$$

#### Tanh function:
$$
\begin{align} 
\frac{d}{dx} f(x) &= \frac{d}{dx} \Bigg[ \frac{sinh(x)}{cosh(x)} \Bigg] \\
&= \Bigg[ \frac{\bigg(\frac{d}{dx}sinh(x)\bigg)cosh(x) - \bigg(\frac{d}{dx}cosh(x)\bigg)sinh(x)}{(cosh(x))^2} \Bigg] && \text{by \eqref{eq6}} \\
& = \frac{(cosh(x))^2 - (sinh(x))^2}{(cosh(x))^2} && \text{by \eqref{eq10} and \eqref{eq11}} \\ 
& = 1 - \bigg(\frac{sinh(x)}{cosh(x)}\bigg)^2 \\
& = 1 - (tanh(x))^2 && \text{by \eqref{eq9}} \\
& = 1 - (f(x))^2 \\
\end{align}
$$

## References:
1. Tensorflow playground [link](http://playground.tensorflow.org)
2. http://cs231n.github.io