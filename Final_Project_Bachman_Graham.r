#############################################
#                                           #
# Author:     Graham Bachman                #
# Date:       02/03/2020                    #
# Subject:    Final Project                 #
# Class:      DSCI 512                      #
# Section:    01W                           #         
# Instructor: Yuanjin Liu                   #
# File Name:  FinalProject_Bachman_Graham.R #
#                                           # 
#############################################


########################
# 1.  Data Preparation #
########################
sum(is.na(insurance))

#     a.  Load the dataset insurance.csv into memory.
library(readr)
insurance <- read_csv("C:/Users/bachm/OneDrive - Maryville University/Fourth Year/SP2020/DSCI 512 - Predictive Modeling/Week 8/insurance.csv")
#make sure sex is factor in memory if it isn't, convert it
if(!is.factor(insurance$sex))
  insurance$sex <- as.factor(insurance$sex)
#make sure smoker is factor in memory if it isn't, convert it
if(!is.factor(insurance$smoker))
  insurance$smoker <- as.factor(insurance$smoker)
#make sure region is factor in memory if it isn't, convert it
if(!is.factor(insurance$region))
  insurance$region <- as.factor(insurance$region)
View(insurance)

#     b.  In the data frame, transform the variable charges by seting
#         insurance$charges = log(insurance$charges). Do not transform
#         it outside of the data frame.
insurance <- insurance[complete.cases(insurance),]
insurance$charges = log(insurance$charges)
#     c.  Using the data set from 1.b, use the model.matrix() function
#         to create another data set that uses dummy variables in place
#         of categorical variables. Verify that the first column only has
#         ones (1) as values, and then discard the column only after
#         verifying it has only ones as values.
insurance.mm <- model.matrix(~ age + sex + bmi + children + smoker + region + charges, data = insurance)
print(insurance.mm)
insurance.mm <- insurance.mm[,-c(1)]
insurance.mm <- as.data.frame(insurance.mm)
#     d.  Use the sample() function with set.seed equal to 1 to generate
#         row indexes for your training and tests sets, with 2/3 of the
#         row indexes for your training set and 1/3 for your test set. Do
#         not use any method other than the sample() function for
#         splitting your data.
set.seed(1)
index = sample(1:nrow(insurance), (2/3)*nrow(insurance))
#     e.  Create a training and test data set from the data set created in
#         1.b using the training and test row indexes created in 1.d.
#         Unless otherwise stated, only use the training and test
#         data sets created in this step.
insurance.train = insurance[index,]
insurance.test = insurance[-index,]
#     f.  Create a training and test data set from data set created in 1.c
#         using the training and test row indexes created in 1.d
insurance.mm.train = insurance.mm[index,]
insurance.mm.test = insurance.mm[-index,]
print(insurance.mm.train)
#################################################
# 2.  Build a multiple linear regression model. #
#################################################

#     a.  Perform multiple linear regression with charges as the
#         response and the predictors are age, sex, bmi, children,
#         smoker, and region. Print out the results using the
#         summary() function. Use the training data set created in
#         step 1.e to train your model.
lm.insurance = lm(charges ~ age + sex + bmi + children + smoker + region, data = insurance.train)
summary(lm.insurance)
#     b.  Is there a relationship between the predictors and the
#         response?
# Yes because the p-value is very close to zero
#     c.  Does sex have a statistically significant relationship to the
#         response?
# Yes because the p-value is less than 0.05
#     d.  Perform best subset selection using the stepAIC() function
#         from the MASS library, choose best model based on AIC. For
#         the "direction" parameter in the stepAIC() method, set
#         direciton="backward"
library(MASS)
lm.aic = stepAIC(lm.insurance, direction = "backward")
lm.aic
# The best model has 8 predictors, age, sexmale, bmi, children, smokeryes, regionnorthwest, regionsoutheast and regionsouthwest

#     e.  Compute the test error of the best model in #3d based on AIC
#         using LOOCV using trainControl() and train() from the caret
#         library. Report the MSE by squaring the reported RMSE.
library(caret)
library(ggplot2)
library(lattice)
train_control = trainControl(method = "LOOCV")
error.model.insurance = train(charges ~ age + sex + bmi + children + smoker + region, data = insurance.train)
print(error.model.insurance)
0.3809761^2
# MSE = 0.1451428

#     f.  Calculate the test error of the best model in #3d based on AIC
#         using 10-fold Cross-Validation. Use train and trainControl
#         from the caret library. Refer to model selected in #3d based
#         on AIC. Report the MSE.
train_control_k <- trainControl(method= "CV", number = 10)
model_k <- train(charges ~ age + sex + bmi + children + smoker + region, data = insurance.train, trControl = train_control, method="lm")
print(model_k)
0.4282153^2
# MSE 0.1833683

#     g.  Calculate and report the test MSE using the best model from 
#         2.d and the test data set from step 1.e.
error.model.test = train(charges ~ age + sex + bmi + children + smoker + region, data = insurance.test)
print(error.model.test)
lm.mse <- round(0.4552151^2,4)
print(lm.mse)
#     h.  Compare the test MSE calculated in step 2.f using 10-fold
#         cross-validation with the test MSE calculated in step 2.g.
#         How similar are they?
# Pretty similar, about 0.02 difference
######################################
# 3.  Build a regression tree model. #
######################################

#     a.  Build a regression tree model using function tree(), where
#         charges is the response and the predictors are age, sex, bmi,
#         children, smoker, and region.
library(tree)
tree.insurance <- tree(charges ~ age + sex + bmi + children + smoker + region, data = insurance.train)
summary(tree.insurance)
#     b.  Find the optimal tree by using cross-validation and display
#         the results in a graphic. Report the best size.
cv.insurance <- cv.tree(tree.insurance)
plot(cv.insurance$size, cv.insurance$dev, ttpe = 'b')
# The best size is 4.

#     c.  Justify the number you picked for the optimal tree with
#         regard to the principle of variance-bias trade-off.
# I would select 4 nodes because the test error is significantly lower than 1, 2 and 3 nodes. Going up to 5 or 6 nodes does further reduce the test error,
# but not as significantly as earlier steps, and because using more nodes runs the risk of overfitting on training data, 4 nodes remains the best pick.
# Using 4 nodes gives us the smallest test error for the fewest number of nodes. 

#     d.  Prune the tree using the optinal size found in 3.b
prune.insurance <- prune.tree(tree.insurance, best = 4)

#     e.  Plot the best tree model and give labels.
plot(prune.insurance)
text(prune.insurance, pretty = 0)

#     f.  Calculate the test MSE for the best model.\
insurance.testV <- unlist(insurance.test)
yhat.insurance <- predict(prune.insurance, newdata = insurance.test)
test.error <- insurance[insurance.testV, "charges"]
tree.mse <- round(mean((yhat.insurance-test.error$charges)^2),4)
print(tree.mse)
####################################
# 4.  Build a random forest model. #
####################################

#     a.  Build a random forest model using function randomForest(),
#         where charges is the response and the predictors are age, sex,
#         bmi, children, smoker, and region.
library(randomForest)
rf.insurance <- randomForest(charges ~ age + sex + bmi + children + smoker + region, data = insurance.train, importance = TRUE)
#     b.  Compute the test error using the test data set.
yhat.rf.insurance <- predict(rf.insurance, newdata = insurance.test)
rf.mse <- round(mean((yhat.rf.insurance - test.error$charges)^2),4)
print(rf.mse)
#     c.  Extract variable importance measure using the importance()
#         function.
importance(rf.insurance)
#     d.  Plot the variable importance using the function, varImpPlot().
#         Which are the top 3 important predictors in this model?
varImpPlot(rf.insurance)
print('The most important variables are smoker, age, and children (or bmi if were are gaughing by purity instead of MSE)')
############################################
# 5.  Build a support vector machine model #
############################################

#     a.  The response is charges and the predictors are age, sex, bmi,
#         children, smoker, and region. Please use the svm() function
#         with radial kernel and gamma=5 and cost = 50.
library(e1071)
svm.insurance = svm(charges~ age + sex + bmi + children + smoker + region, data = insurance.train, kernel = "radial", gamma = 5, cost = 50)
#     b.  Perform a grid search to find the best model with potential
#         cost: 1, 10, 50, 100 and potential gamma: 1,3 and 5 and
#         potential kernel: "linear","radial" and
#         "sigmoid". And use the training set created in step 1.e.
insurance.tune = tune(svm, charges ~ age + sex + bmi + children + smoker + region, data = insurance.train,
                      ranges = list(kernel = c("linear","radial","sigmoid"),cost = c(1, 10, 50, 100), gamma = c(1, 3, 5)))
#     c.  Print out the model results. What are the best model
#         parameters?
summary(insurance.tune)
print('The best parameters are the kernel radial, a cost of 1 and a gamma of 1')
#     d.  Forecast charges using the test dataset and the best model
#         found in c).
pred.charges = predict(insurance.tune$best.model, newdata = insurance.test)
#     e.  Compute the MSE (Mean Squared Error) on the test data.
insurance.test <- unlist(insurance.test)
true.observe = insurance[insurance.test, "charges"]
print(pred.charges)
table(true.observe$charges, pred.charges)
svm.mse <- round(1.34334, 4)

#############################################
# 6.  Perform the k-means cluster analysis. #
#############################################

#     a. Remove the sex, smoker, and region, since they are not numerical values.
numeric.insurance <- insurance[-c(2,5,6)]
print(numeric.insurance)
library(cluster)
library(factoextra)
#     b.  Determine the optimal number of clusters, and use the
#         gap_stat method and set iter.max=20. Justify your answer.
#         It may take longer running time since it uses a large dataset.
fviz_nbclust(numeric.insurance, kmeans, method = "gap_stat")
print('Optimal number of clusters is 2')
#     d.  Perform k-means clustering using the optimal number of
#         clusters found in step 6.c. Set parameter nstart = 25
km.insurance <- kmeans(numeric.insurance, 2, nstart=25)
#     e.  Visualize the clusters in different colors, setting parameter
#         geom="point"
fviz_cluster(km.insurance, data = numeric.insurance)

######################################
# 7.  Build a neural networks model. #
######################################

#     a.  Remove the sex, smoker, and region, since they are not numerical values.
print(numeric.insurance)
library(neuralnet)
#     b. Standardize the inputs using the scale() function.
scaled.insurance = scale(numeric.insurance)
#     c. Convert the standardized inputs to a data frame using the as.data.frame() function.
scaled.insurance <- as.data.frame(scaled.insurance)
#     d. Split the dataset into a training set containing 80% of the original data and the test set containing the remaining 20%.
index = sample(1:nrow(scaled.insurance), 0.8*nrow(scaled.insurance))
scaled.insurance.train = scaled.insurance[index,]
scaled.insurance.test = scaled.insurance[-index, ]
#     e. The response is charges and the predictors are age, bmi, and children. Please use 1 hidden layer with 1 neuron.
insurance.nn = neuralnet(charges ~ age + bmi + children, data = scaled.insurance.train, hidden = c(1),linear.output = FALSE)
?neuralnet
#     f.  Plot the neural network.
plot(insurance.nn)
#     g.  Forecast the charges in the test dataset.
predict.insurance.nn = compute(insurance.nn, scaled.insurance.test[,c("age", "bmi", "children")])
#     h. Get the observed charges of the test dataset.
observe.insurance.test = scaled.insurance.test$charges
#     d.  Compute test error (MSE).
nn.mse <- round(mean((observe.insurance.test - predict.insurance.nn$net.result)^2),4)
print(nn.mse)
################################
# 8.  Putting it all together. #
################################

#     a.  For predicting insurance charges, your supervisor asks you to
#         choose the best model among the multiple regression,
#         regression tree, random forest, support vector machine, and
#         neural network models. Compare the test MSEs of the models
#         generated in steps 2.g, 3.f, 4.b, 5.e, and 7.d. Display the names
#         for these types of these models, using these labels:
#         "Multiple Linear Regression", "Regression Tree", "Random Forest", 
#         "Support Vector Machine", and "Neural Network" and their
#         corresponding test MSEs in a data.frame. Label the column in your
#         data frame with the labels as "Model.Type", and label the column
#         with the test MSEs as "Test.MSE" and round the data in this
#         column to 4 decimal places. Present the formatted data to your
#         supervisor and recommend which model is best and why.

Model.Type <- c("Multiple Linear Regression", "Regression Tree", "Random Forest", "Support Vector Machine", "Neural Network")
Test.MSE <- c(lm.mse, tree.mse, rf.mse, svm.mse, nn.mse)
data.frame(Model.Type, Test.MSE)
# -- The Multiple Linear Regression Model is the best because it has the lowest Test MSE.
#    Which means the model is not overfitted to training data and does a good job of outputting
#    the correct output when given new data')

#     b.  Another supervisor from the sales department has requested
#         your help to create a predictive model that his sales
#         representatives can use to explain to clients what the potential
#         costs could be for different kinds of customers, and they need
#         an easy and visual way of explaining it. What model would
#         you recommend, and what are the benefits and disadvantages
#         of your recommended model compared to other models?

# --   A random forest with a variable imp plot stands out as an ideal choice because,
#      the plot is easy to understand visually and does a good job of demonstrating, 
#      what factors or costs will contribute the most to a response. We can make the response,
#      customer expenses and then easily show how different customer attributes affect cost.,
#      One disadvantage is model accuracy (based on the results of the calculated test error).,
#      The random forest is not our most accurate model. One advantage that a regression tree
#      might have over the random forest model is an even easier to understand visual. Decision trees/
#      dendograms are very easy to interpret and do not require much mathematical explanation. A 
#      Regression tree may be a preferable choice, depending on how diverse our customer base is and how many
#      cases we want to show.')

#     c.  The supervisor from the sales department likes your regression
#         tree model. But she says that the sales people say the numbers
#         in it are way too low and suggests that maybe the numbers
#         on the leaf nodes predicting charges are log transformations
#         of the actual charges. You realize that in step 1.b of this
#         project that you had indeed transformed charges using the log
#         function. And now you realize that you need to reverse the
#         transformation in your final output. The solution you have
#         is to reverse the log transformation of the variables in 
#         the regression tree model you created and redisplay the result:
#

unlogged.tree <- prune.insurance
unlogged.tree$frame
unlogged.tree$frame[c("yval")] <- lapply(unlogged.tree$frame[c("yval")], exp)
plot(unlogged.tree)
text(unlogged.tree, pretty = 0)