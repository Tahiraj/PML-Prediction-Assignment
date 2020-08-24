# PML-Prediction-Assignment
Peer-graded Assignment: Prediction Assignment Writeup

### Introduction

#### Problem
One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants.

### Objective
The goal of this project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set that we use with other variables to predict with. 

```{r}
set.seed(12345)
library(caret)
```

## Download training and testing data and cleaning the data

```{r}
url_train <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
training <- read.csv(url_train, na.strings = c("", "NA", "#DIV/0!"))
url_test <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
testdata<- read.csv(url_test, na.strings = c("", "NA", "#DIV/0!"))
dim(training )
dim(testdata)
# removing "NA" columns
table(colSums(is.na(training)))
colselect <- colnames(training)[colSums(is.na(training)) == 0]
colselect
# First 7 columns do not relate to excercise
colselect<- colselect[8: length(colselect)]
training <- training[colselect]
```

### Create training and test sets
70% for training and 30% validation
```{r}
inTrain = createDataPartition(training$classe, p = 0.7, list=FALSE)
testing = training[-inTrain,]
training = training[ inTrain,]
dim(training)
dim(testing)
training$classe <- factor(training$classe)
testing$classe<- factor(testing$classe)
nzv = nearZeroVar(x = training)
nzv
```

We use different model to predict and choose the one with highest accuracy
### Predicting with trees

```{r}
fitRpart <- train(classe ~., data=training, method="rpart")
print(fitRpart$finalModel)
predRpart<- predict(fitRpart, testing)
levels(testing$classe)
levels(predRpart)
confusionMatrix(testing$classe, predRpart)
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1525   29  116    0    4
##          B  484  385  270    0    0
##          C  499   37  490    0    0
##          D  423  187  354    0    0
##          E  153  159  289    0  481
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4895          
##                  95% CI : (0.4767, 0.5024)
##     No Information Rate : 0.524           
##     P-Value [Acc > NIR] : 1               
##                                           
##                   Kappa : 0.3324          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.4945  0.48306  0.32258       NA  0.99175
## Specificity            0.9468  0.85181  0.87723   0.8362  0.88870
## Pos Pred Value         0.9110  0.33802  0.47758       NA  0.44455
## Neg Pred Value         0.6298  0.91319  0.78823       NA  0.99917
## Prevalence             0.5240  0.13543  0.25811   0.0000  0.08241
## Detection Rate         0.2591  0.06542  0.08326   0.0000  0.08173
## Detection Prevalence   0.2845  0.19354  0.17434   0.1638  0.18386
## Balanced Accuracy      0.7206  0.66743  0.59991       NA  0.94023
#fancyRpartPlot(fitRpart$finalModel)
```

### Random forests
```{r}
fitRf <- train(classe ~ ., data = training, method = "rf")
predRf <- predict(fitRf, testing)
confusionMatrix(testing$classe, predRf)
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1672    1    1    0    0
##          B    8 1129    2    0    0
##          C    0    6 1017    3    0
##          D    0    0    7  956    1
##          E    0    0    1    1 1080
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9947          
##                  95% CI : (0.9925, 0.9964)
##     No Information Rate : 0.2855          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9933          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9952   0.9938   0.9893   0.9958   0.9991
## Specificity            0.9995   0.9979   0.9981   0.9984   0.9996
## Pos Pred Value         0.9988   0.9912   0.9912   0.9917   0.9982
## Neg Pred Value         0.9981   0.9985   0.9977   0.9992   0.9998
## Prevalence             0.2855   0.1930   0.1747   0.1631   0.1837
## Detection Rate         0.2841   0.1918   0.1728   0.1624   0.1835
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9974   0.9959   0.9937   0.9971   0.9993
```

### Boosted trees 
```{r}
fitGbm <- train(classe ~., data=training, method="gbm", verbose=FALSE)
predGbm<- predict(fitGbm, testing)
confusionMatrix(testing$classe, predGbm)
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1638   25    3    8    0
##          B   40 1063   31    4    1
##          C    0   33  981   11    1
##          D    0    2   34  920    8
##          E    2   11    6    9 1054
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9611          
##                  95% CI : (0.9558, 0.9659)
##     No Information Rate : 0.2855          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9508          
##                                           
##  Mcnemar's Test P-Value : 1.171e-05       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9750   0.9374   0.9299   0.9664   0.9906
## Specificity            0.9914   0.9840   0.9907   0.9911   0.9942
## Pos Pred Value         0.9785   0.9333   0.9561   0.9544   0.9741
## Neg Pred Value         0.9900   0.9850   0.9848   0.9935   0.9979
## Prevalence             0.2855   0.1927   0.1793   0.1618   0.1808
## Detection Rate         0.2783   0.1806   0.1667   0.1563   0.1791
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9832   0.9607   0.9603   0.9787   0.9924
```
### Linear Discriminant Analysis (lda) 

```{r}
fitLda <- train(classe ~ ., data = training, method = "lda")
predLda <- predict(fitLda, testing)
confusionMatrix(testing$classe, predLda)
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1397   40  110  123    4
##          B  170  738  128   47   56
##          C  111  112  655  127   21
##          D   59   36  121  720   28
##          E   41  195   92  103  651
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7071          
##                  95% CI : (0.6952, 0.7187)
##     No Information Rate : 0.3021          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6289          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.7857   0.6583   0.5922   0.6429   0.8566
## Specificity            0.9326   0.9158   0.9224   0.9488   0.9159
## Pos Pred Value         0.8345   0.6479   0.6384   0.7469   0.6017
## Neg Pred Value         0.9095   0.9193   0.9072   0.9187   0.9773
## Prevalence             0.3021   0.1905   0.1879   0.1903   0.1291
## Detection Rate         0.2374   0.1254   0.1113   0.1223   0.1106
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.8591   0.7871   0.7573   0.7958   0.8862
```
### Out of Sample error

### Accuracy

```{r}
print(paste0("RPART accuracy = ", confusionMatrix(predRpart, testing$classe)$overall['Accuracy']))
## [1] "RPART accuracy = 0.489549702633815"
print(paste0("RF accuracy = ", confusionMatrix(predRf, testing$classe)$overall['Accuracy']))
## [1] "RF accuracy = 0.994732370433305"
print(paste0("GBM accuracy = ", confusionMatrix(predGbm, testing$classe)$overall['Accuracy']))
## [1] "GBM accuracy = 0.961087510620221"
print(paste0("LDA accuracy = ", confusionMatrix(predLda, testing$classe)$overall['Accuracy']))
## [1] "LDA accuracy = 0.707051826677995"
```

Random forest has the highiest accuracy, so we use for predicting the test data 

```{r}
testRf <- predict(fitRf, testdata)
testRf
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
