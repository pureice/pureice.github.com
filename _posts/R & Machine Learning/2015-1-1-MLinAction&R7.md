---
layout: post
title: 机器学习实战 & R：7、AdaBoost
comments: true
category: R & Machine Learning
tags: [机器学习, R, AdaBoost]
---

本系列博客为入门级博客，如果你初次接触R或者机器学习，欢迎收看；如果你已经入门，那么可以自己搜索资料学习。本系列博客为书籍《机器学习实战》的R语言实现版本，原书上是使用Python实现的，此系列博客主要包括每一部分的机器学习技术的简述、R实现的方法。以下为每一部分的链接，如果到了主页，说明还没有写：

<!-- more -->
- [1、前言与资源](https:/pureice.github.io/blog/2014/12/17/MLinAction&R1.html)
- [2、K-近邻算法](https:/pureice.github.com)
- [3、决策树](https:/pureice.github.com)
- [4、朴素贝叶斯](https:/pureice.github.com)
- [5、Logistic回归](https:/pureice.github.com)
- [6、支持向量机](https:/pureice.github.com)
- [7、AdaBoost](https:/pureice.github.comblog/2015/1/1/MLinAction&R7.html)
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
- [1. AdaBoost方法简介](#1. 方法简介)
- [2. R实现单决策树的AdaBoost](#2. R实现)
- [3. 其他分类相关问题](#3. 相关问题)
<!-- /MarkdownTOC -->

<a name="1. 方法简介" />

###1. 方法简介

Meta-algrithm(元算法)的基本思想是不听从一个人的意见，而是听从多人的意见。AdaBoost是元算法的其中一种，也有人说AdaBoost是最好的监督学习算法。

这一章以单层决策树作为弱学习机，使用AdaBoost作为集合方法，将拥有不同样本（这里的不同样本其实是一个样本，看了下面便知道原因）生成不同的弱学习机（同一类型，不同参数的单层决策树）结合生成一个强学习机。

####1.1 基于数据集多重抽样的分类器

将多种分类器结合在一起的方法就是*集成方法*(Ensemble Method)或者*元算法*(Meta-algrithm)，这种集合可以是：不同算法集成；同一算法不同设置下的继集成；数据集中不同部分分配给不同分类其之后的集成。

AdaBoost的优缺点

> 优点：泛化错误率低，已编码，可以用在大部分分类器上，无参数调节；
> 缺点：对离群值敏感；
> 数据类型：数值型和标称型数据。

#####1.1.1 Bagging方法

Bagging算法又称自举汇聚法(Booststrap aggregating)。Bagging法的特点在于抽样，抽样的方法为重抽样，例如一个样本中，有N条记录，重抽样便是在这N条记录中抽取N条记录，在R中可以使用sample(x, size, replace = True)的方法进行重抽样。重抽样得到S个新的样本，每个样本便可以生产一个弱学习机，便是S个弱学习机，最后的结果便是这S个弱学习机的集成方法，例如S个分类器中结果对多的那个。Bagging方法不仅可以用于分类，同样也使用于计算统计量，例如计算一组N个数样本的均值，如果怀疑这个样本的可靠性，可以使用Bagging生成S个样本，也可以生成S个均值，可以得到这个均值的均值和方差，有文献报道在够大的抽样情况下，这N个样本的均值的均值和方差可以收敛到一个稳定的值。

还有一些先进Bagging法，例如随即森林[(Random Forest)](http://en.wikipedia.org/wiki/Random_forest)，还是看英文的好。

#####1.1.2 Boosting方法

Boosting和Bagging方法相似，也是用的同样类型的学习机。Boosting方法更关注的是已有分类器错分的那些数据，最后结果也是不同弱分类器的分类效能加权而得，分类效能低的权重低，分类效能高的权重高；而Bagging最后结果靠不同弱分类的权重是一样的。

Yoav Freund在1996的Experiments with a New Boosting Algorithm中提出了AdaBoost.M1和AdaBoost.M2两种算法。其中，AdaBoost.M1是我们通常所说的Discrete AdaBoost：而AdaBoost.M2是M1的泛化形式。该文的一个结论是:当弱分类器算法使用简单的分类方法时，boosting的效果明显地统一地比bagging要好。当弱分类器算法使用C4.5时，boosting比bagging较好，但是没有前者的比较来得明显。

Boosting中最流行的是AdaBoost，两者的区别在于：Boosting需要得到每个分类器的最小效能，即弱学习机的先验知识；AdaBoost则没有这个缺点。

####1.2 AdaBoost的一般流程

> 1. 收集数据：可以使用任意方法；
> 2. 准备数据：依赖于所使用的弱分类器类型，本章使用的是单层决策树，这种分类器可以以处理任何数据类型。当然也可以使用任意分类器作为弱分类器，第2章到第6章中的任一分类器都可以充当弱分类器。作为 弱分类器，简单分类器的效果更好；
> 3. 分析数据：可以使用任意方法；
> 4. 训练算法：AdaBoost的大部分时间都用在训絲上，分类器将多次在同一数据集上训练弱分类器；
> 5. 测试算法：计算分类的错误率，或者说还有其他的测试方法，例如正确率、召回率和ROC曲线(计算其AUC)；
> 6. 使用方法：同SVM样，AdaBoost预测两个类别中的一个。如果想把它应用到多个类别的场合，那么就要像多类SVM中的做法一样AdaBoost进行修改 。

####1.3 AdaBoost理论介绍

AdaBoost是adaptive Boosting即（自适应Boosting）的缩写，其运行过程如下：训练样本中的每个记录， 并赋予其一个权重，这些权重构成了向量D。迭代初始，这些权重都初始化成相等值。首先在训练数据上训练出一个弱分类器并计算该分类器的错误率，然后在同一数据集上再次训练弱分类器。在分类器的第二次训练当中，将会重新调整每个样本的权重，其中第一次分对的样本的权重将会降低，而第一次分错的样本的权重将会提高，这里的权重将会用于这一次训练中错误率的计算，得到加权错误率。为了从所有弱分类器中得到最终的分类结果，AdaBoost为每个分类器都分配了一个权重值alpha，这些alpha值是基于每个弱分类器的错误率进行计算的。最后对于新的数据集进行算法使用时，将会用所有得到的弱分类器进行计算，结果是这些弱分类器结果加权后的和。

错误率的定义是：$$\epsilon= \frac{未正确分类的样本数目}{所有样本数目}$$。

alpha的计算公式如下：$$\alpha =0.5 * ln(\frac{1-\epsilon}{\epsilon})$$。

在计算出alpha后，样本的权重向量*D*进行更新：1. 如果分类正确，$$D_i^{(i+1)}=\frac{D_i^{(t)}*e^{-\alpha}}{Sum(D)}$$，权重降低；
2. 如果分类错误，$$D_i^{(i+1)}=\frac{D_i^{(t)}*e^{\alpha}}{Sum(D)}$$，权重增加。

在计算出新的*D*之后，新的迭代开始；关于上面这几个公式的推到，可以参考AdaBoost[原始文献](http://link.springer.com/chapter/10.1007/3-540-59119-2_166)。

AdaBoost的大致流程图如下：

<img src="http://pureice.github.com/images/ML/7-1.JPG" height="50%" width="50%">

<a name="2. R实现" />

###2. R实现单决策树的AdaBoost

具体的每个函数的意义见原文，与原文完全相同，只是使用R并根据R的一些数据函数特点进行了修改。

####2.1 单层决策树

首先生成数据：

    loadSimpData <- function(){
    datMat <- matrix(data=c(1,2,1.3,1,2,2.1,1.1,1,1,1),nrow=5,ncol=2)
    classLables <- matrix(data=c(1,1,-1,-1,1),nrow=1,ncol=5)
    return(list(data1=datMat,data2=classLables))#注意R中返回多个值时的方法
    }
    datMat=loadSimpData()[[1]]#注意R中调用list中具体元素的方法，两个[]
    classLables=loadSimpData()[[2]]

这个函数生产两个数据，一个是datMat：所有feature的数据；一个是classlables：数据对应的分类标签。

决策树的具体内容见第三章：第三章用的ID3方法，决策树生产的方法还有CART和C4.5方法。接下来构建单层决策树，单层决策树是指只通过一个feature的进行判断分类，当然这是一个很不严谨的分类方法，但是也是一个很好理解、很容易解释的弱学习机。单层决策树的code如下

	stumpClassify <- function(dataMatrix,dimen,threshVal,threshIneq){
	  dimen=as.numeric(dimen)  #加上这3个数据转化，是为了蛋疼的R数据结构
	  threshVal=as.numeric(threshVal)
	  threshIneq=as.character(threshIneq)
	  retArray <- matrix(data=1,nrow=dim(dataMatrix)[1],ncol=1)#开始假设所有的都分对了
	  if (threshIneq == 'lt'){
	    retArray[dataMatrix[,dimen] <= threshVal] <- -1 #retArray[这里面的返回值是F和T]，如果是小于为假，那么小于阈值的为假
	  }else{
	    retArray[dataMatrix[,dimen] > threshVal] <- -1 #如果是大于，那么大于阈值为假
	  }
	  return(retArray)#返回此次阈值测试的结果
	}

	buildStump <- function(dataArr,classLabels,D){
	  dataMatrix=as.matrix(dataArr);labelMat=t(as.matrix(classLables))
	  m=dim(dataMatrix)[1] #行数
	  n=dim(dataMatrix)[2] #列数
	  numSteps=10.0 #步数
	  bestStump=list() #Python中使用的是字典，我们这里使用list
	  bestClasEst=matrix(data=0,nrow=m,ncol=1) #最好的分类结果
	  minError=Inf #inf是不行的，Python里面可以，但是R里面只有Inf才可以，注意大小写
	  for (i in seq(1,n)){#按照有变量个数进行测试，这里数据只有两个
	    rangMin=min(dataMatrix[,i]);rangMax=max(dataMatrix[,i]) #测试阈值的最大与最小
	    stepSize=(rangMax-rangMin)/numSteps
	    for (j in seq(-1,as.integer(numSteps)+1)){#这里的步长选择了-1到11，阈值在最大最小之外，是可以的。
	      for (inequal in c("lt","gt")){#这里面的lt表示小于，gt表示大于，大于小于的意思就是决策树里面的左拐还是右拐
	        threshVal=rangMin + j*stepSize
	        predictedVals <- stumpClassify(dataMatrix,i,threshVal,inequal)
	        errArr <- matrix(data=1,nrow=m,ncol=1)
	        errArr[predictedVals == labelMat]=0
	        weightedArr <- sum(t(D)*errArr)#这个地方和Python中不一样，Python中直接求和了，R中没有，需要加一个
	        if (weightedArr < minError){
	          minError = weightedArr
	          bestClasEst = predictedVals #原来Python中用的copy函数，这是因为Python中拷贝的问题，R中没有这个问题
	          bestStump$dim = i
	          bestStump$thresh = threshVal
	          bestStump$ineq = inequal
	        }
	      }
	    }
	  }
	  return(list(bestStump,minError,bestClasEst))
	}

单层决策树的实现共有两个函数完成，第一个函数通过阈值比较把数据进行分类，返回相应的结论：例如以第dimen个变量的阈值threshVal，如果ineq为lt，即little than（gt表示greater than），那边表示的是这个feature的值如果小于阈值，那么返回结果为否；第二个事变例函数，便利所有feature中的所有值，选择出最好的单层决策树，让error最小。测试这两个函数

	#若分类器结果测试
	sampleNum = dim(datMat)[1]
	D=matrix(data=1/sampleNum,nrow=1,ncol=sampleNum) #这个和原文有点出入
	result=buildStump(datMat,classLables,D)

	> result
	[[1]]  
	[[1]]$dim #选择哪一个feature
	[1] 1

	[[1]]$thresh #这个feature的阈值
	[1] 1.3

	[[1]]$ineq #表示小于这个阈值为负
	[1] "lt"


	[[2]] 
	[1] 0.2 #平均误差，总误差除以样本个数，例如这5个分类中只有一个发生		  #错误，那么就是说总误差为1，1/5=0.2

	[[3]]   #表示分类的结果
	     [,1]
	[1,]   -1 #只有这个是错误的剩下的都是对的
	[2,]    1
	[3,]   -1
	[4,]   -1
	[5,]    1

	#结果检查：clear！

####2.2 AdaBoost实现

接下来实现AdaBoost，其伪代码如下：

	对每次迭代：
		利用buildStump()函数找到最佳的单层决策树
		将最佳单层决策树加入到单层决策树数组
		计算alpha
		计算新的权重向量D
		更新累计类别估计值
		如果错误率等于0.0,则退出循环

其R代码如下：

	adaBoostTrainDS <- function(dataArr,classLables,numIt=40){
	  dataArr=as.matrix(dataArr)
	  classLables=as.matrix(classLables)
	  weakClassArr=list() #Python中用字典，我们依然用list
	  m=dim(dataArr)[1]
	  D=matrix(data=1/m,nrow=1,ncol=5)
	  aggClassEst=matrix(data=0,nrow=m,ncol=1)
	  for (i in seq(1,numIt)){
	    result=buildStump(datMat,classLables,D)#这个就是R里面比Python笨的地方
	    bestStump=result[[1]]
	    error=result[[2]]
	    classEst=result[[3]]
	    alpha=0.5*log((1.0-error)/max(error,1e-16))
	    bestStump$alpha=alpha#在这里面R与Python的不同，觉得要简单点，Python的[]用的是append 
	    expon=-1*alpha*classLables*t(classEst)#这个地方是R简单点的地方，R中要求两个相乘的vector方向一致，multipy是一对一的相乘
	    D=D*exp(expon)#这个地方到底是谁乘以谁有点烦躁
	    D=D/sum(D)
	    aggClassEst = aggClassEst + alpha*classEst
	    aggErrors = matrix(data=1,nrow=1,ncol=5)
	    aggErrors[sign(aggClassEst) == t(classLables)] = 0
	    errorRate=sum(aggErrors)/m
	    #result$errorRate=errorRate
	    weakClassArr[[paste0(i,"thclassfier")]]=bestStump#蛋疼的R数据结构啊啊啊啊
	    if (errorRate <= 0){
	      break
	    }
	  }
	  return(weakClassArr)
	}

测试Code：

	result2=adaBoostTrainDS(datMat,classLables)
	> result2
	$`1thclassfier`  #第一个分类器
	$`1thclassfier`$dim
	[1] 1

	$`1thclassfier`$thresh
	[1] 1.3

	$`1thclassfier`$ineq
	[1] "lt"

	$`1thclassfier`$alpha
	[1] 0.6931472


	$`2thclassfier`
	$`2thclassfier`$dim
	[1] 2

	$`2thclassfier`$thresh
	[1] 1

	$`2thclassfier`$ineq
	[1] "lt"

	$`2thclassfier`$alpha
	[1] 0.9729551


	$`3thclassfier`
	$`3thclassfier`$dim
	[1] 1

	$`3thclassfier`$thresh
	[1] 0.9

	$`3thclassfier`$ineq
	[1] "lt"

	$`3thclassfier`$alpha
	[1] 0.8958797

	#检验结果：clear！

结果显示，共生成了3个弱分类器，他们的结合为一个强学习机，错误率为0。

####2.3 测试算法

虽然生成了强学习机，并且训练的错误率为0，但是还需要使用测试一下其效果。

	adaClassify <- function(datToClass,classifierArr){
	  dataMatrix=datToClass
	  m=dim(dataMatrix)[1]
	  aggClassEst=matrix(data=0,nrow=m,ncol=1)
	  for (i in seq(length(classifierArr))){
	    classEst=stumpClassify(dataMatrix,classifierArr[[i]]["dim"],classifierArr[[i]]["thresh"],classifierArr[[i]]["ineq"])
	    aggClassEst=aggClassEst+as.numeric(classifierArr[[i]]["alpha"])*classEst
	  }
	  return(sign(aggClassEst))
	}
	

测试结果为：

	datToClass=matrix(data=c(5,0,5,0),nrow=2,ncol=2) #蛋疼的数据结构，注意一行是一个数据
	result3=adaClassify(datToClass,result2)

	> result3
	     [,1]
	[1,]    1
	[2,]   -1

	#结果检查：clear！

结果显示成功的将(5,5)，(0,0)两个数据进行了分类

<a name="3. 相关问题" />

###3. 其他分类相关问题

####3.1 过拟合问题

<img src="http://pureice.github.com/images/ML/7-2.JPG" height="50%" width="50%">

观察表中的测试错误率一栏，就会发现测试错误率在达到了一个最小值之后又开始上升了。这类现象称之为过拟合（overfitting, 也称为过学习）。有文献声称，对于表现好的数据集，AdaBoost的测试错误率就会达到一个稳定值，并不会随着分类器的增多而上升。或许在本例子中的数据集也称不上“表现好”。该数据集一开始有30%的缺失值，对于Logistic回归而言，这些缺失值的假设就是有效的，而对于决策树却可能并不合适。 

很多人都认为 ，AdaBoost和SVM是监督机器学习中最强大的两种方法。实际上， 这两者之间拥有不少相似之处。我们可以把弱分类器想象成SVM中的一个核函数，也可以按照最大化某个最小间隔的方式重写AdaBoost。而它们的不同在于其所定义的间隔计算方式有所不同，因此导致的结果也不同。特别是在高维空间下，这两者之间的差异就会更加明显。

####3.2 其他度量方式

除了本章使用的最简单的错误率，还有一些其他度量方式，如正确率和召回率，以下面的混淆矩阵为例

<img src="http://pureice.github.com/images/ML/7-3.JPG" height="50%" width="50%">

其中正确率(Precision)为TP/(TP+FP)，表示预测为正确中真的是正确的比例，召回率(Recall)为TP/(TP+FN)，表示所有真的是正确中有多少被预测为正确了。

对于分类问题还有一个很General的图，[ROC曲线](http://baike.baidu.com/view/42249.htm)图，如下图，

<img src="http://pureice.github.com/images/ML/7-4.JPG" height="50%" width="50%">

其中横坐标为假阳率=FP/(FP+TN)，纵坐标的真阳率=TP/(TP+FN)，一个好的分类方法如上图棕红色线。量化评价ROC曲线的方法是计算曲线下面积(Area Under the Curve, AUC)，完美的分类器AUC=1，完全随机分类的ROC曲线的AUC=0.5。

由于R和Python画图的方法不太一样，所以这里不重复ROC曲线和AUC计算的函数，有很多[网络资源](http://blog.sina.com.cn/s/blog_9b332cf401012qht.html)和R的包：[pROC](http://cran.r-project.org/web/packages/pROC/)等。

####3.3 非均衡分类问题与代价函数

我们还必须讨论一个问题，平常状态下我们都假设所有类别的分类代价是一样的 。例如我们构建了一个用于检测患疝病的马匹是否存活的系统在，通过构建了分类器，假如某人给我们牵来一匹马，他希望我们能预测这匹马能否生存，如果我们说马会死，那么他们就可能会对马实施安乐死，而不是通过给马喂药来延缓其不可避免的死亡过程。对于预测应该死亡，我们预测的是不死亡，代价是一些马药，而对于预测不应该死亡却预测的是死亡，代价则是一匹昂贵的马，马药和马这两个代价明显是不一样的，也就是所谓的非均衡分类问题。

我们有一些其他可以用于处理非均匀分类代价问题的方法，其中的一种称为代价敏感的学习(cost-sensitiveleaming)。在分类算法中，我们有很多方法可以用来引入代价信息。在AdaBoost中，可以基于代价函数来调整错误权重向量D。在朴素贝叶斯中，可以选择具有最小期望代价而不是最大概率的类别作为最后的结果。在SVM中，可以在代价函数中对于不同的类别选择不同的参数。上述做法就会给较小类更多的权重，即在训练时，小类当中只允许更少的错误。

针对非均衡问题调节分类器的方法，除了上面的新的评价方法，还有的是对分类器的训练数据进行改造。这可以通过欠抽样(undersampling)或者过抽样(oversampling)来实现。过抽样意味着复制样例，而欠抽样意味着删除样例。不管采用哪种方式，数据都会从原始形式改造为新形式。抽样过程则可以通过随机方式或者某个预定方式来实现。

第7章的Code[下载地址](https://github.com/pureice/pureice.github.com/tree/master/code/ML%26R) and the [Englishe edition](https://github.com/pureice/pureice.github.com/tree/master/code/ML%26R).

如需转载，请著名作者Robin Li以及[Pureice.github.com](http:/pureice.github.cim)，谢谢你的配合~