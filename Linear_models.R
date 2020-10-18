library(MASS) # for ridge
library(glmnet) 
train=read.csv(file="train3.csv", header=TRUE)
test=read.csv(file="test3.csv", header=TRUE)
test=subset(test, select = -c(duration) )



#RMSE using LM in python = 4.121860459310065
train2=subset(train, select = c(age,sex_3,sex_4,sex_5,country_1,country_3,country_4,country_5,country_7,country_8,country_9,country_10,country_12,V1_2,V1_3,V1_4,V1_5,V1_6,V1_7,V1_8,V1_10,V1_11,dry.mouth,expectoration,Asymptomatic,runny.nose,rhinorrhea,malaise,chest.pains,throat.problems,coughs,muscle.pain,nausea,shortness.difficulty.breathing,diarrhea,headache,fatigue,cold.chills,duration))
test2=subset(test, select = c(age,sex_3,sex_4,sex_5,country_1,country_3,country_4,country_5,country_7,country_8,country_9,country_10,country_12,V1_2,V1_3,V1_4,V1_5,V1_6,V1_7,V1_8,V1_10,V1_11,dry.mouth,expectoration,Asymptomatic,runny.nose,rhinorrhea,malaise,chest.pains,throat.problems,coughs,muscle.pain,nausea,shortness.difficulty.breathing,diarrhea,headache,fatigue,cold.chills))
#RMSE using LM in python = 4.138137680774762
train3=subset(train, select = c(age,sex_4,sex_5,country_1,country_3,country_4,country_5,country_7,country_8,country_9,country_10,country_12,V1_2,V1_3,V1_4,V1_5,V1_6,V1_7,V1_8,V1_10,V1_11,dry.mouth,expectoration,Asymptomatic,rhinorrhea,malaise,chest.pains,throat.problems,coughs,nausea,shortness.difficulty.breathing,diarrhea,headache,fatigue,cold.chills,duration))
test3=subset(test, select = c(age,sex_4,sex_5,country_1,country_3,country_4,country_5,country_7,country_8,country_9,country_10,country_12,V1_2,V1_3,V1_4,V1_5,V1_6,V1_7,V1_8,V1_10,V1_11,dry.mouth,expectoration,Asymptomatic,rhinorrhea,malaise,chest.pains,throat.problems,coughs,nausea,shortness.difficulty.breathing,diarrhea,headache,fatigue,cold.chills))
#RMSE using LM in python = 4.136431693568705
train4=subset(train, select = c(sex_4,sex_5,country_1,country_3,country_4,country_5,country_7,country_8,country_9,country_10,country_12,V1_2,V1_3,V1_5,V1_6,V1_7,V1_8,V1_10,V1_11, dry.mouth,expectoration,Asymptomatic, rhinorrhea,malaise,chest.pains,throat.problems,coughs,nausea,shortness.difficulty.breathing,diarrhea,headache,fatigue,cold.chills,duration))
test4=subset(test, select = c(sex_4,sex_5,country_1,country_3,country_4,country_5,country_7,country_8,country_9,country_10,country_12,V1_2,V1_3,V1_5,V1_6,V1_7,V1_8,V1_10,V1_11, dry.mouth,expectoration,Asymptomatic, rhinorrhea,malaise,chest.pains,throat.problems,coughs,nausea,shortness.difficulty.breathing,diarrhea,headache,fatigue,cold.chills))
#RMSE using LM in python = 4.1358905404856205
train5=subset(train, select = c(sex_4,sex_5,country_1,country_3,country_4,country_5,country_7,country_8,country_9,country_10,country_12,V1_2,V1_3,V1_5,V1_6,V1_7,V1_8,V1_10,V1_11,dry.mouth,expectoration,Asymptomatic,rhinorrhea,malaise,chest.pains,throat.problems,coughs,nausea,diarrhea,headache,fatigue,cold.chills,duration))
test5=subset(test, select = c(sex_4,sex_5,country_1,country_3,country_4,country_5,country_7,country_8,country_9,country_10,country_12,V1_2,V1_3,V1_5,V1_6,V1_7,V1_8,V1_10,V1_11,dry.mouth,expectoration,Asymptomatic,rhinorrhea,malaise,chest.pains,throat.problems,coughs,nausea,diarrhea,headache,fatigue,cold.chills))
#RMSE using LM in python = 4.019270792139617
train6=subset(train, select = c(sex_4,country_1,country_3,country_4,country_5,country_8,country_10,country_12,V1_2,V1_3,V1_5,V1_6,V1_8,V1_10,V1_11,dry.mouth,Asymptomatic,rhinorrhea,chest.pains,throat.problems,coughs,diarrhea,fatigue,cold.chills,duration))
test6=subset(test, select = c(sex_4,country_1,country_3,country_4,country_5,country_8,country_10,country_12,V1_2,V1_3,V1_5,V1_6,V1_8,V1_10,V1_11,dry.mouth,Asymptomatic,rhinorrhea,chest.pains,throat.problems,coughs,diarrhea,fatigue,cold.chills))
#RMSE using LM in python = 4.093402725907847
train7=subset(train, select = c(sex_4,country_1,country_4,country_5,country_8,country_10,country_12,V1_2,V1_3,V1_5,V1_6,V1_8,V1_10,V1_11, dry.mouth,Asymptomatic,rhinorrhea,chest.pains,throat.problems,coughs,diarrhea,cold.chills,duration))
test7=subset(test, select = c(sex_4,country_1,country_4,country_5,country_8,country_10,country_12,V1_2,V1_3,V1_5,V1_6,V1_8,V1_10,V1_11, dry.mouth,Asymptomatic,rhinorrhea,chest.pains,throat.problems,coughs,diarrhea,cold.chills))
#RMSE using LM in python = 4.1284916891379435
train8=subset(train, select = c(sex_4,country_1,country_4,country_5,country_8,V1_2,V1_3,V1_6, dry.mouth,Asymptomatic,chest.pains,throat.problems,coughs,diarrhea,duration))
test8=subset(test, select = c(sex_4,country_1,country_4,country_5,country_8,V1_2,V1_3,V1_6, dry.mouth,Asymptomatic,chest.pains,throat.problems,coughs,diarrhea))
#RMSE using LM in python = 4.15476459286247
train9=subset(train, select = c(sex_4,country_1,country_4,country_5,country_8,V1_2,V1_3,V1_6,dry.mouth,Asymptomatic,throat.problems,coughs,diarrhea,duration))
test9=subset(test, select = c(sex_4,country_1,country_4,country_5,country_8,V1_2,V1_3,V1_6,dry.mouth,Asymptomatic,throat.problems,coughs,diarrhea))
#RMSE using LM in python = 4.066897285064696
train10=subset(train, select = c(sex_4,country_1,country_5,country_8,V1_2,V1_3,V1_6,dry.mouth,Asymptomatic,throat.problems,coughs,diarrhea,duration))
test10=subset(test, select = c(sex_4,country_1,country_5,country_8,V1_2,V1_3,V1_6,dry.mouth,Asymptomatic,throat.problems,coughs,diarrhea))
#RMSE using LM in python = 4.135382648957618
train11=subset(train, select = c(sex_4,country_1,country_5,country_8,V1_2,V1_3,V1_6,dry.mouth,Asymptomatic,throat.problems,coughs,duration))
test11=subset(test, select = c(sex_4,country_1,country_5,country_8,V1_2,V1_3,V1_6,dry.mouth,Asymptomatic,throat.problems,coughs))




set.seed(1)
#LASSO
X.LASSO.train<- as.matrix(subset(train7, select = -c(duration)))
X.LASSO.test<- as.matrix(test7)
Y.train=train7[,"duration"]

cv.lasso.1 <- cv.glmnet(y=Y.train, x= X.LASSO.train, family="gaussian")
cv.lasso.1
plot(cv.lasso.1) # Plot CV-MSPE
coef(cv.lasso.1) # Print out coefficients at optimal lambda 
coef(cv.lasso.1, s=cv.lasso.1$lambda.min) # Another way to do this.
# Using the "+1SE rule" (see later) produces a sparser solution
coef(cv.lasso.1, s=cv.lasso.1$lambda.1se) # Another way to do this.

# Predict both halves using first-half fit
pred.las1.min <- predict(cv.lasso.1, newx=X.LASSO.test, s=cv.lasso.1$lambda.min)
pred.las1.1se <- predict(cv.lasso.1, newx=X.LASSO.test, s=cv.lasso.1$lambda.1se)



#LM 
LM=lm(duration~.,data=train9)
pred.LM=predict(LM, test9)



#Ridge
lambda.vals =seq(0, 200, .05)
ridge <- lm.ridge(duration~., lambda = lambda.vals, data=train8)

ind.min.GCV = which.min(ridge$GCV)
lambda.min = lambda.vals[ind.min.GCV]
lambda.min 
# Show coefficient path
plot(ridge)
select(ridge)
(coef.ri.best = coef(ridge)[which.min(ridge$GCV),])

pred.ri = as.matrix(cbind(1,test3)) %*% coef.ri.best



#random forest
library(randomForest)
set.seed(1)
rf=randomForest(duration~age+coughs+throat.problems, data=train)
library(caret)
pred.rf=predict(rf,test)

varImpPlot(rf,sort=T,n.var=20)

#####################################################################
###               LASSO and LM Cross validation                   ###
#####################################################################
### Number of folds
K = 5
### Let's define a function for constructing CV folds
get.folds = function(n, K) {
  ### Get the appropriate number of fold labels
  n.fold = ceiling(n / K) # Number of observations per fold (rounded up)
  fold.ids.raw = rep(1:K, times = n.fold) # Generate extra labels
  fold.ids = fold.ids.raw[1:n] # Keep only the correct number of labels
  
  ### Shuffle the fold labels
  folds.rand = fold.ids[sample.int(n)]
  
  return(folds.rand)
}
get.MSPE = function(Y, Y.hat){
  return(mean((Y - Y.hat)^2))
}

### Construct folds
n = nrow(train) # Sample size
folds = get.folds(n, K)

### Create a container for MSPEs. Let's include ordinary least-squares
### regression for reference
all.models = c("LM" ,"LASSO-Min", "LASSO-1se")
all.MSPEs = array(0, dim = c(K, length(all.models)))
colnames(all.MSPEs) = all.models

### Begin cross-validation
for(i in 1:K){
  ### Split data
  data.train = train11[folds != i,]
  data.valid = train11[folds == i,]
  X.LASSO.train<- as.matrix(subset(data.train, select = -c(duration)))
  X.LASSO.valid<- as.matrix(subset(data.valid, select = -c(duration)))
  n.train = nrow(X.LASSO.train)
  
  ### Get response vectors
  Y.train = data.train$duration
  Y.valid = data.valid$duration
  
  cv.lasso.1 <- cv.glmnet(y=Y.train, x= X.LASSO.train, family="gaussian")

  # Predict both halves using first-half fit
  pred.las1.min <- predict(cv.lasso.1, newx=X.LASSO.valid, s=cv.lasso.1$lambda.min)
  pred.las1.1se <- predict(cv.lasso.1, newx=X.LASSO.valid, s=cv.lasso.1$lambda.1se)
  
  MSPE.LASSO.min = get.MSPE(Y.valid, pred.las1.min)
  all.MSPEs[i, "LASSO-Min"] = sqrt(MSPE.LASSO.min)
  
  MSPE.LASSO.1se = get.MSPE(Y.valid, pred.las1.1se)
  all.MSPEs[i, "LASSO-1se"] = sqrt(MSPE.LASSO.1se)
  
  LM1=lm(duration~.,data=data.train)
  pred.LM=predict(LM1, data.valid)
  MSPE.LM=get.MSPE(Y.valid,pred.LM)
  all.MSPEs[i, "LM"] = sqrt(MSPE.LM)
}
this.ave.MSPEs = apply(all.MSPEs, 2, mean)


#output the text file for submission.
duration=pred.rf
ID=seq(1:200)
pred1=data.frame(ID,duration)
colnames(pred1)=c("ID","duration")
write.table(pred1, file = "kaggle_submission_CZ.txt", 
            sep = ",",row.names = FALSE, col.names =TRUE)

