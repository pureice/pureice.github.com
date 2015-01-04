#Machine Learning in Action & R
#Application of adaboost
#2014.12.31
#Robin-Li：http://pureice.github.com
#Code in UTF-8

#################################
######Data
#################################
loadSimpData <- function(){
  datMat <- matrix(data=c(1,2,1.3,1,2,2.1,1.1,1,1,1),nrow=5,ncol=2)
  classLables <- matrix(data=c(1,1,-1,-1,1),nrow=1,ncol=5)
  return(list(data1=datMat,data2=classLables))#care for the method that return multi-type data in a function
}
datMat=loadSimpData()[[1]]#Calling the definite element in a list should use double [ ]
classLables=loadSimpData()[[2]]

##################################
#Weak classifier: Sigle layer decision stump
###################################
stumpClassify <- function(dataMatrix,dimen,threshVal,threshIneq){
  dimen=as.numeric(dimen)  #Because of the form of data, it is useful to transfer the data type.
  threshVal=as.numeric(threshVal)
  threshIneq=as.character(threshIneq)
  retArray <- matrix(data=1,nrow=dim(dataMatrix)[1],ncol=1)#It assumed that all result is true
  if (threshIneq == 'lt'){
    retArray[dataMatrix[,dimen] <= threshVal] <- -1 #retArray[return is T ot F]. 
    #If using 'lt', it means the value little than threshval will return F (we define the ineq means inequal)
  }else{
    retArray[dataMatrix[,dimen] > threshVal] <- -1 
  }
  return(retArray)
}

buildStump <- function(dataArr,classLabels,D){
  dataMatrix=as.matrix(dataArr);labelMat=t(as.matrix(classLables))
  m=dim(dataMatrix)[1] #number of row
  n=dim(dataMatrix)[2] #number of colunm
  numSteps=10.0 #number of step
  bestStump=list() #The original code in Python use data of dictinary. However, we use list in R
  bestClasEst=matrix(data=0,nrow=m,ncol=1) #the best classfing result
  minError=Inf #inf is not working in R, which in ok in python. The R will check the uppercase and lowercase letters.
  for (i in seq(1,n)){#alternating of the number of featur and there is just two features in this code
    rangMin=min(dataMatrix[,i]);rangMax=max(dataMatrix[,i]) #max and min value of a sigle feature
    stepSize=(rangMax-rangMin)/numSteps
    for (j in seq(-1,as.integer(numSteps)+1)){#It should be noted that the range of threshvalue if out of the max and min vlaue. That is ok! 
      for (inequal in c("lt","gt")){#lt means "litter than" and gt means "greater than"
        threshVal=rangMin + j*stepSize
        predictedVals <- stumpClassify(dataMatrix,i,threshVal,inequal)
        errArr <- matrix(data=1,nrow=m,ncol=1)
        errArr[predictedVals == labelMat]=0
        weightedArr <- sum(t(D)*errArr)#It should be noted the difference of the function in Python and R
        if (weightedArr < minError){
          minError = weightedArr
          bestClasEst = predictedVals #The origlal code in Python use the copy() which is the result of the memory sharing of the variate in Python
          bestStump$dim = i
          bestStump$thresh = threshVal
          bestStump$ineq = inequal
        }
      }
    }
  }
  return(list(bestStump,minError,bestClasEst))
}

#Test of weak classifer
sampleNum = dim(datMat)[1]
D=matrix(data=1/sampleNum,nrow=1,ncol=sampleNum)
result=buildStump(datMat,classLables,D)

#Result：clear！

############################################
#####Completed application of adaboost
############################################
adaBoostTrainDS <- function(dataArr,classLables,numIt=40){
  dataArr=as.matrix(dataArr)
  classLables=as.matrix(classLables)
  weakClassArr=list() #we still use list in R instead of dictionary in python
  m=dim(dataArr)[1]
  D=matrix(data=1/m,nrow=1,ncol=5)
  aggClassEst=matrix(data=0,nrow=m,ncol=1)
  for (i in seq(1,numIt)){
    result=buildStump(datMat,classLables,D)#That is the difficult in R when return nulti-result in a function
    bestStump=result[[1]]
    error=result[[2]]
    classEst=result[[3]]
    alpha=0.5*log((1.0-error)/max(error,1e-16))
    bestStump$alpha=alpha#Use of list in R
    expon=-1*alpha*classLables*t(classEst)
    D=D*exp(expon)
    D=D/sum(D)
    aggClassEst = aggClassEst + alpha*classEst
    aggErrors = matrix(data=1,nrow=1,ncol=5)
    aggErrors[sign(aggClassEst) == t(classLables)] = 0
    errorRate=sum(aggErrors)/m
    #result$errorRate=errorRate
    weakClassArr[[paste0(i,"thclassfier")]]=bestStump
    if (errorRate <= 0){
      break
    }
  }
  return(weakClassArr)
}

# Test

result2=adaBoostTrainDS(datMat,classLables)

#Result：clear！

#########################################
###Test of AdaBoost algrithm
#########################################
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

#Test

datToClass=matrix(data=c(5,0,5,0),nrow=2,ncol=2) 
result3=adaClassify(datToClass,result2)

#result：clear！