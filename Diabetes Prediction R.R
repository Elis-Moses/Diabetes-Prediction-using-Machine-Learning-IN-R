#Diabetes Prediction using Machine Learning IN R

install.packages('caret')
install.packages('randomForest')
install.packages('pROC')
install.packages('ggcorrplot')
install.packages('car')

# Loading necessary libraries
library(tidyverse)
library(caret)
library(randomForest)
library(pROC)
library(ggcorrplot)
library(car)


# Setting working directory and loading the dataset
setwd("C:/Users/LDO SYSTEMS/Desktop/R Data Sci")
Data <- read.csv("diabetesData.csv")
head(Data)
View(Data)

# Checking the structure and summary of the data
str(Data)
summary(Data)

#checking for missing values
colSums(is.na(Data))

# Data Preprocessing - Replacing zero values in specific columns with NA
cols_to_fix <- c("plasma_glucose_conc", "bp", "tricepsthickness", "insulin", "BMI")
Data[cols_to_fix] <- lapply(Data[cols_to_fix], function(x) ifelse(x == 0, NA, x))

# Imputing missing values using median
Data[cols_to_fix] <- lapply(Data[cols_to_fix], function(x) ifelse(is.na(x), median(x, na.rm = TRUE), x))

# Converting target variable to factor
Data$target <- as.factor(Data$target)

# Splitting data into training (80%) and testing (20%) sets
set.seed(123)
trainIndex <- createDataPartition(Data$target, p = 0.8, list = FALSE)
trainData <- Data[trainIndex, ]
testData <- Data[-trainIndex, ]

view(trainData)
# Training Logistic Regression model
log_model <- train(target ~ ., data = trainData, method = "glm", family = "binomial", trControl = trainControl(method = "cv", number = 10))
summary(log_model)

# Training Random Forest model with hyperparameter tuning
rf_grid <- expand.grid(mtry = c(2, 3, 4))
rf_model <- train(target ~ ., data = trainData, method = "rf", trControl = trainControl(method = "cv", number = 10), tuneGrid = rf_grid)
print(rf_model)

# Making predictions
log_preds <- predict(log_model, testData)
rf_preds <- predict(rf_model, testData)

# Evaluate models
log_cm <- confusionMatrix(log_preds, testData$target)
rf_cm <- confusionMatrix(rf_preds, testData$target)

# Printing accuracy
print(log_cm$overall["Accuracy"])
print(rf_cm$overall["Accuracy"])

# ROC Curve for Random Forest
rf_probs <- predict(rf_model, testData, type = "prob")
roc_rf <- roc(testData$target, rf_probs[,2])

# ROC Curve for Logistic Regression
log_probs <- predict(log_model, testData, type = "prob")
roc_log <- roc(testData$target, log_probs[,2])

# Plot ROC Curves
plot(roc_rf, col = "blue", main = "ROC Curve Comparison")
lines(roc_log, col = "red")
legend("bottomright", legend = c("Random Forest", "Logistic Regression"), col = c("blue", "red"), lwd = 2)

# Print AUC values
auc(roc_rf)
auc(roc_log)

# Feature Importance Visualization
importance <- varImp(rf_model, scale = FALSE)
print(importance)
plot(importance, main = "Feature Importance - Random Forest")

## Fine-tuning the Random Forest Model

# Expanding grid with more hyperparameters
rf_grid <- expand.grid(mtry = c(2, 3, 4, 5, 6)) 

# Training Random Forest with different hyperparameters
rf_model_tuned <- train(
  target ~ ., data = trainData, method = "rf",
  trControl = trainControl(method = "cv", number = 10),
  tuneGrid = rf_grid,
  ntree = 500,  # Increase number of trees
  nodesize = 5  # Control overfitting
)

print(rf_model_tuned)

# Evaluating performance
rf_preds_tuned <- predict(rf_model_tuned, testData)
rf_cm_tuned <- confusionMatrix(rf_preds_tuned, testData$target)
print(rf_cm_tuned$overall["Accuracy"])

## Feature Correlation & Multicollinearity Check

# Computing correlation matrix (exclude target since it's categorical)
cor_matrix <- cor(trainData %>% select(-target))

# Plotting heatmap
ggcorrplot(cor_matrix, method = "circle", type = "lower", lab = TRUE, lab_size = 3)

# Checking Variance Inflation Factor (VIF)
vif_model <- lm(pedigree_func ~ ., data = trainData)
vif_values <- vif(vif_model)

print(vif_values)
