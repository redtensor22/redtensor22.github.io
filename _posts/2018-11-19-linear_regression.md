---
title: "Regression from Scratch"
date: 2018-11-19
tags: [machine learning, data science, neural network]
header:
  image: "/images/linear_regression.jpg"
excerpt: "machine learning"
mathjax: "true"
---

# Project Objective

The purpose behind this project was to dig into the mathematics behind a typical regression algorithm. Once the mathematics were fully grasped, the next step was to write from scratch an original regression algorithm. Finally, the comparison would be made between my personal algorithm and the scikit learn algorithm in Python. 

# The Theory Behind Linear Regression

Imagining a simple problem in which we have input data $$ x\in \mathbb{R}^1$$ and a scalar label set $$y\in\mathbb{R}^1$$, a plot might look like (data generated using 'x, y = make_regression(n_samples=100, n_features=1, noise=10)')

<img src="{{ site.url }}{{ site.baseurl }}/images/random_data.png" alt="Randomly generated data">

In such a plot, it is quite easy to see that the brunt of the data falls along a line which migh be described using a simple funcion such as $$y=mx+b$$. In the arena of data science, we would refer to the variables in this equation as $$w_0$$ and $$w_1$$, also known as the bias and slope. The goal in a linear regression calculation would be to find a function, $$\mathcal{f}(x)$$, which maps the domain data, $$x$$, to the output data, $$y$$. This function is called the regression function and has two parameters (for the case of 1D data) to be determined, namely, the slope and bias. 

For input data generalized to $$d$$ dimensional data, the equation to solve becomes $$y_i \approx \mathcal{f}(x_i) = w_0 + \sum_{j=1}^{d}x_{ij}w_j$$. The problem is now set up. The next step to take is to find an objective function which provides us with satisfactory values of the slope and bias. The function we will use for this is the least squares function:

$$w = argmin \sum_{i=1}^{n}(y_i-\mathcal{f}(x_i))^2 $$

What we are doing is summing over each data point and at each data point finding the difference between the actual truth value and the value predicted by our regression function. Minimizing this error will give us satisfactory values for the hidden variables $$w_0$$ and the slope. 

The formula above holds, however can be simplified using vector notation along with linear algebra. One bit of information which will prove important during coding this will be extension of the input data by one dimension to include bias. We can write a $$d$$ dimensional data point as 

$$x^T = \begin{bmatrix} 1,&x_{i1},&x_{i2},&...&x_{id}\\ \end{bmatrix}$$

Because each of these dimensions will have their own slope, we can also write a vector of $$w$$'s as 

$$w = \begin{bmatrix} w_0,&w_{1},&w_{2},&...&w_{d}\\ \end{bmatrix}$$

We now want to find the maximum likelyhood of the least squares function with respect to vector $$w$$. This can be done using the gradient (for n data points)

$$\sum_{i}^{n}\nabla_w(y_i^2-2w^Tx_iy_i + w^Tx_ix_i^Tw) = 0$$

which can be solved to give  

$$ w_{LS} = (\sum_{i=1}^{n}x_ix_i^T)^{-1}(\sum_{i=1}^{n}y_ix_i)$$

And easy extension of this notation is moving into matrix form. Following the exact same procedure, we can derive the result

$$ w_{LS} = (X^TX)^{-1}(X^Ty)$$

Where y is now a vector, and X is a matrix of dimension $$n\times(d+1)$$. New data can then easily be found using the relation $$y_{pred} \approx x_{new}^Tw_{LS} $$



