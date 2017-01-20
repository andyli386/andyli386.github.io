---
title: Generalized Linear Models
date: 2016-01-13 22:47:02
tags: ml
---


---
博客地址：[52ml.me](http://www.52ml.me)<br>
原创文章，版权声明：自由转载-非商用-非衍生-保持署名 | [Creative Commons BY-NC-ND 3.0](http://creativecommons.org/licenses/by-nc-nd/3.0/deed.zh)
<br>
---
<!--- 
本人初学机器学习，为了掌握*sklearn*开始翻译该文档，如有不妥之处还请指正。


## 1.1. 广义线性模型 [Generalized Linear Models](http://scikit-learn.org/stable/modules/linear_model.html#generalized-linear-models)

The following are a set of methods intended for regression in which the target value is expected to be a linear combination of the input variables. In mathematical notion, if $\hat{y}$ is the predicted value. 
\\[
\hat{y}(w, x) = w_0 + w_1 x_1 + ... + w_p x_p
\\]
Across the module, we designate the vector w = (w_1,
..., w_p) as coef_ and w_0 as intercept_.
To perform classification with generalized linear models, see Logistic regression.

以下方法用于回归，回归的预测值为输入变量的线性组合。在数学概念中如果$\hat{y}$是预测值，那么
\\[
\hat{y}(w, x) = w_0 + w_1 x_1 + ... + w_p x_p
\\]
在这个模块中，我们指定向量$w = (w_1, ..., w_p)$作为系数，$w_0$作为截距。
使用广义线性模型进行*分类*，见*[Logistic回归](http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)*。
### 1.1.1. 普通最小二乘法 [Ordinary Least Squares](http://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares)

LinearRegression fits a linear model with coefficients $w = (w_1, ..., w_p)$ to minimize the residual sum of squares between the observed responses in the dataset, and the responses predicted by the linear approximation. Mathematically it solves a problem of the form:
$$\underset{w}{min\,} {|| X w - y||_2}^2$$
![plot_ols_0011.png](http://scikit-learn.org/stable/_images/plot_ols_0012.png)
<img src="http://scikit-learn.org/stable/_images/plot_ols_0012.png" width="60%" height="50%" align=center>
LinearRegression will take in its fit method arrays X, y and will store the coefficients w of the linear model in its coef_ member:

线性回归拟合一个系数为$w = (w_1, ..., w_p)$的线性模型，然后最小化数据集中的观测值和该线性模型预测值的残差平方和。数学上它可以解决以下形式的问题：
$$\underset{w}{min\,} {|| X w - y||_2}^2$$
<img src="http://scikit-learn.org/stable/_images/plot_ols_0012.png" width="60%" height="50%" align=center>

[LinearRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression)将数组X，y传到已经拟合的函数中，并将线性模型的系数w存储在coef_成员变量中:

```python
>>> from sklearn import linear_model
>>> clf = linear_model.LinearRegression()
>>> clf.fit ([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
>>> clf.coef_
array([ 0.5,  0.5])
```

However, coefficient estimates for Ordinary Least Squares rely on the independence of the model terms. When terms are correlated and the columns of the design matrix X have an approximate linear dependence, the design matrix becomes close to singular and as a result, the least-squares estimate becomes highly sensitive to random errors in the observed response, producing a large variance. This situation of multicollinearity can arise, for example, when data are collected without an experimental design.
--->