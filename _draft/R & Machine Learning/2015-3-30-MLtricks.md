---
layout: post
title: 机器学习中的一些tricks
comments: true
category: R & Machine Learning
tags: [机器学习, tricks]
---

关于怎么选择features和怎么选择model的tricks。

<!-- more -->

这个是[原始地址](http://t.cn/RA2ofBt?u=2310909444&m=3824331790717045&cu=2310909444)，以及机器学习中不应该犯得[错误](http://ml.posthaven.com/machine-learning-done-wrong)

这篇文章主要是解决featture selection和model selection的tricks。当有10个features的时候怎么办，或者当有1000个features时怎么办；是选择logistic 回归好还是随即森林好。

##1、feature selection和data exploration

在进行模型拟合之前选择合适的features是非常重要的内容。

在这之间需要对数据进行浏览：哪些数据是可用的；哪些数据与目标是相关的；features之间的相关性有什么用处；是否有偏度，是否有较大的拖尾，没有这个拖尾会不会效果好点；探索3倍IQR外的数据生成的原因是什么。

如果数据量极大，那么尝试从中选出一部分数据进行实验。

同时我们可能还需要进行以下的操作：

- 1、将连续变量变为离散变量;
- 2、将features进行结合的话可能效果更好；
- 3、将变量平砍或者立方；
- 4、forward selection；
- 5、backward selection；
- 6、当有很多features都有用的时候，使用PCA或者SVD。

##2、model selection

模型的选择很大程度上取决于数据的类型。

- logistic 回归：y是二分类，有大量数据可以train。
- Naive Bayes：多分类结果，有限的数据。
- Decision Tree：容易理解，不用管是线性还是非线性的。但是容易过拟合。
- Random Forests：使用Bootstrap技术的DT，可以解决overfit的问题。
- SVM：极高的准确性，但是计算有问题，过高的memory。
- 神经网络：不需要很多统计训练。但是最大的区别在于难以解释。

选择哪一个模型的一个方法是bias-variance trade off。选择80-20k-fold的 validation进行trade off。如果variance高，那么增加train data或者减少features；如果bias高，那么增加features或者换features。

另外，对于二分类问题，ROC曲线是一个非常有用的方法；还有一个方法是计算精确率和召回率。

##3、不应该犯的错误

这些错误说起来简单，但是在显示应用当中就经常会忘记。

1、使用了错误的损失函数：要根据显示情况使用，即时调整损失函数，比如说不均衡酚类问题。
2、使用线性函数来代表非线性的内容：尤其是当遇到分类问题的时候直接上logistic回归，但是logistic却是线性的。

3、对于离群值的态度。有时候会删除离群值，但是离群值有时候可能会代表了一些重要信息；另外，一些算法对于离群值很敏感，例如adaboost。

4、当n远小于p的时候，可能会造成严重的overfit问题。

5、在使用正则项的时候忘记对features标准化，因为正则项是相加的，大的会吃掉小的。

6、使用线性模型拟合多y情况时，忘记y的共线性问题。

7、依然是共线性的问题，x的共线性问题，解决这个问题可以删除一些features。

如需转载，请著名作者Robin Li以及[Pureice.github.com](http:/pureice.github.cim)，谢谢你的配合~