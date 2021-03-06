---
layout: post
title: 机器学习实战 & R：1、前沿与资源
comments: true
category: R & Machine Learning
tags: [机器学习, R, 前沿与资源]
---

本系列博客为入门级博客，如果你初次接触R或者机器学习，欢迎收看；如果你已经入门，那么可以自己搜索资料学习。本系列博客为书籍《机器学习实战》的R语言实现版本，原书上是使用Python实现的，此系列博客主要包括每一部分的机器学习技术的简述、R实现的方法。以下为每一部分的链接，如果到了主页，说明还没有写：

<!-- more -->
- [1、前言与资源](https:/pureice.github.io/blog/2014/12/17/MLinAction&R1.html)
- [2、K-近邻算法](https:/pureice.github.com)
- [3、决策树](https:/pureice.github.com)
- [4、朴素贝叶斯](https:/pureice.github.com)
- [5、Logistic回归](https:/pureice.github.com)
- [6、支持向量机](https:/pureice.github.com)
- [7、AdaBoost](https:/pureice.github.com/blog/2015/1/1/MLinAction&R7.html)
- [8、回归](https:/pureice.github.com)
- [9、树回归](https:/pureice.github.com)
- [10、K-均值聚类](https:/pureice.github.com)
- [11、Apriori算法关联分析](https:/pureice.github.com)
- [12、FP-grouth算法](https:/pureice.github.com)
- [13、PCA主成份分析](https:/pureice.github.com)
- [14、SVD奇异值分析](https:/pureice.github.com)
<!-- more -->

###目录
<!-- MarkdownTOC depth=4 -->
- [1. 前言](#1. 前言)
- [2. 机器学习的任务](#2. 任务)
- [3. 资源](#3. 资源)
<!-- /MarkdownTOC -->

<a name="1. 前言" />

###1. 前言

####1.1 关于本系列博客

本博客是《机器学习实战》这本书的R语言博客，原书的实现方式的Python。

<img src="http://pureice.github.com/images/ML/1-1.jpg" height="70%" width="70%">

关于这本书的内容可以在[豆瓣](http://book.douban.com/subject/24703171/)里面看到。所使用的数据可以在[CSDN](http://download.csdn.net/detail/mctyro/6504521)中下载到。

本系列博客以原书为模版，每一章为一个博客，也就是每一个博客包含一个机器学习技术，主要包括每个技术的基本概念和R实现的方法。

####1.2 什么是机器学习和R

机器学习是什么呢？如果你已了解机器学习为了学习R，跳过这一部分；如果你已了解R为了学习机器学习，那么可以在这里找到，[百度百科](http://baike.baidu.com/link?url=ITVsjYdM0ltZ2c2yNDskUn9tAtepZKjnKMlaLQhFb6Nvwjdh4oPm4-pfdyWPSFzpfE7MeylrvoGRSjCgesVZFq)和[WikiPedia](http://en.wikipedia.org/wiki/Machine_learning)。

至于什么是R，也可以在[百度百科](http://baike.baidu.com/subview/30957/10711822.htm#viewPageContent)和[R的主页](http://www.r-project.org/)找到。总体来说，R是统计学家发明的一个科学计算开源语言，随着数据科学的快速发展，R的运用已经爆炸式在各个行业：金融、生命科学、科研、公共卫生等开花结果。

<a name="2. 任务" />

###2. 任务

####2.1 监督学习与非监督学习

什么是监督学习与非监督学习，这个是初次接触机器学习的人很可能困惑的问题。

从概念上讲：

    有监督学习：对具有概念标记（分类）的训练样本进行学习，以尽可能对训练样本集外的数据进行标记（分类）预测。这里，所有的标记（分类）是已知的。因此，训练样本的岐义性低。

	无监督学习：对没有概念标记（分类）的训练样本进行学习，以发现训练样本集中的结构性知识。这里，所有的标记（分类）是未知的。因此，训练样本的岐义性高。聚类就是典型的无监督学习。

从数据结构上讲，以线性函数的数据结构为例：
  $$ y = \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n + \gamma $$.
如果数据中，存在y，那么就是监督学习；没有y的存在，那么就是非监督学习。

####2.2 算法分类

在机器学习算法中，国际权威的学术组织the IEEE International Conference on Data Mining (ICDM) 2006年12月评选出了数据挖掘领域的十大经典算法：C4.5, k-Means, SVM, Apriori, EM, PageRank, AdaBoost, kNN, Naive Bayes, and CART。一张图表现这些算法，同时，这些算法的分类也是机器算法的基本分类。

<img src="http://pureice.github.com/images/ML/1-2.jpg" height="70%" width="70%">

在实际运用中，对于这些算法的选择，主要本着两个原则：1）数据的结构与类型；2）运用的目的是什么。一个实际运用的流程主要包括以下几点：

1. 收集数据与数据预处理：这一部分可以是整个运用中最重要的一部分，一般会占用所有时间中的80%，初学者一定要以200%的精力关注这个部分；
2. 分析数据：以所运用的目的为主，分析要使用的数据结构，选择可以使用的算法；
3. 训练算法：使用数据来估计算法中的参数，数据一般会分为训练集和测试集，训练集就是做这一部分的内容；
4. 测试算法：那么测试集就是测试几个备选算法中哪个算法最好，测试集就是做这一部分的内容；
5. 使用算法：使用算法得到结论。

要注意的是这样一句话：没有最好的算法，只有最合适的算法。千万不要去寻找最好的算法，每一个现实的内容，每一种类型的数据结构与类型，最合适的算法都不同；最合适的算法是试出来的，不是找出来的。

<a name="3. 资源" />

###3. 资源

####3.1 所需初始知识

* 线性代数：之所以需要学习线性代数，因为在计算机处理大量数据的时候一般使用的是矩阵格式，同时很多算法的数学描述格式也是矩阵格式；学习线性代数可以帮助更进一步的了解机器学习算法的本质，同时也可以提升编程的效率。

* 统计概率知识：现在很多算法都是基于概率论的算法，因此学习统计概率知识是必须的。

* 简单的计算机知识：这个知识就不用说了，如果你学过C之类的(VB不算。因为我见过很多学习VB之后依然很难上手R的)，那么恭喜，R作为一种很像伪代码的语言，你很快就能上手了。

####3.2 理论知识的学习与使用资料

#####网络资源

* 其实已经有很多人对资源进行了总结，转载一个MachineLearner的[博客内容](http://v.163.com/special/opencourse/daishu.html)

* 进阶的机器学习内容，[UFLDL](http://deeplearning.stanford.edu/wiki/index.php/UFLDL%E6%95%99%E7%A8%8B)

* 好友[Chrispher](http://www.datakit.cn/)的博客

#####公开课程

* 最先推荐的肯定是Coursera上[Andrew Ng的机器学习](www.coursera.org)

* 网易的斯坦福大学公开课：[机器学习](http://v.163.com/special/opencourse/machinelearning.html)

* 线性代数的公开课，麻省理工的[Gilbert Strang](http://v.163.com/special/opencourse/daishu.html)的线性代数，真的很赞。

* 统计学相关的课程还是强烈推荐来自[Coursera的课程](www.coursera.org)

####3.3 R的学习与使用资料

#####网络资源

* [人大经济论坛](http://bbs.pinggu.org/forum-69-1.html)上面关于R的内容有很多，是很推荐的。

* 如果你学习R的时候发现有问题，那么去[stackoverflow](http://stackoverflow.com/)上找，90%你问的问题都有。

* 如果你是生命科学、生物信息学的使用者，并且不知道[Bioconductor](http://www.bioconductor.org/)这个名字，那么现在需要知道了；其中的[论坛](https://support.bioconductor.org/)很值得关注。

* 很推荐的一个博主，尤其适合生命科学的人，糗糗的[糗世界](http://pgfe.umassmed.edu/ou/)。

* 博主[大鹏志的菜鸟入门教程](http://dapengde.com/r4dummies/)

#####图书

* [R语言实战](http://book.douban.com/subject/20382244/)

* 推荐的书就这么一本，这本书入门够了。当你入门之后，你会发现，最好的书就是Help，学会看Help，那么就不需要什么书了。