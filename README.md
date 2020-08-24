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
#fancyRpartPlot(fitRpart$finalModel)
```

### Random forests
```{r}
fitRf <- train(classe ~ ., data = training, method = "rf")
predRf <- predict(fitRf, testing)
confusionMatrix(testing$classe, predRf)
```

### Boosted trees 
```{r}
fitGbm <- train(classe ~., data=training, method="gbm", verbose=FALSE)
predGbm<- predict(fitGbm, testing)
confusionMatrix(testing$classe, predGbm)
```
### Linear Discriminant Analysis (lda) 

```{r}
fitLda <- train(classe ~ ., data = training, method = "lda")
predLda <- predict(fitLda, testing)
confusionMatrix(testing$classe, predLda)
```
### Out of Sample error

### Accuracy

```{r}
print(paste0("RPART accuracy = ", confusionMatrix(predRpart, testing$classe)$overall['Accuracy']))
print(paste0("RF accuracy = ", confusionMatrix(predRf, testing$classe)$overall['Accuracy']))
print(paste0("GBM accuracy = ", confusionMatrix(predGbm, testing$classe)$overall['Accuracy']))
print(paste0("LDA accuracy = ", confusionMatrix(predLda, testing$classe)$overall['Accuracy']))
```

Random forest has the highiest accuracy, so we use for predicting the test data 

```{r}
testRf <- predict(fitRf, testdata)
testRf
```
