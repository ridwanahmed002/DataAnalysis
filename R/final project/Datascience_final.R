
library(tidyverse)
library(skimr)
library(e1071)
library(purrr)
library(caTools)
library(naivebayes)
library(caret)

sample_data <- read_csv("/home/ridwan/Documents/aiub/data science/final project/Churn_Modelling.csv")

head(sample_data)


str(sample_data)


skim_without_charts(sample_data)


apply(sample_data, MARGIN = 2, FUN = function(x) sum(is.na(x)))


sum(duplicated(sample_data))


plot(sample_data$HasCrCard, sample_data$IsActiveMember)

boxplot(sample_data$Balance  , main= "Boxplot of Balance " , boxwex=0.1)

boxplot(sample_data$Age  , main= "Boxplot of Age " , boxwex=0.1)

ggplot(data = sample_data) +
  geom_bar(mapping = aes(x = Geography, fill= Gender))

sample_data <- sample_data %>%
  mutate(
    Surname = as.factor(Surname),
    Geography = as.factor(Geography),
    Gender = as.factor(Gender)
  )

categorical_vars <- c("Surname", "Geography", "Gender")

chi_squared_test <- function(var) {
  chisq <- chisq.test(sample_data[[var]], sample_data$Exited)
  p_value <- chisq$p.value
  return(data.frame(Variable = var, P_Value = p_value))
}


chi_squared_results <- map_df(categorical_vars, chi_squared_test)

significant_attributes <- chi_squared_results %>%
  filter(P_Value < 0.05) %>%
  pull(Variable)


print(significant_attributes)

sample_data2 <- subset(sample_data, select = -c(RowNumber, CustomerId, Surname, Gender))

head(sample_data2)

sample_data2$Loyalty <- sample_data2$Tenure/sample_data2$Age

head(sample_data2)

sample_data2$Geography <-  as.integer(factor(sample_data2$Geography))
head(sample_data2)

barplot(prop.table(table(sample_data2$Exited)),
        col = rainbow(2),
        ylim = c(0, 1),
        main = "Class Distribution")

sample_data3 <- subset(sample_data2, select = -c(Age, Tenure))
head(sample_data3)

X <- sample_data3 %>%
  select(-Exited) 
y <- sample_data3$Exited


set.seed(123)
train_indices <- createDataPartition(y, p = 0.7, list = FALSE)
train_data <- sample_data3[train_indices, ]
test_data <- sample_data3[-train_indices, ]


nb_model <- naiveBayes(Exited ~ ., data = train_data, laplace = 1)  # Set laplace parameter (smoothing) to 1 or another appropriate value


predicted_test <- predict(nb_model, newdata = test_data)

test_data$Exited <- factor(test_data$Exited, levels = levels(predicted_test))


accuracy_test <- confusionMatrix(predicted_test, test_data$Exited)$overall['Accuracy']
print(accuracy_test)


accuracy_values <- numeric(10)


folds <- cut(seq(1, nrow(sample_data3)), breaks = 10, labels = FALSE)


for (i in 1:10) {

  test_indices <- which(folds == i)
  test_data <- sample_data3[test_indices, ]
  train_data <- sample_data3[-test_indices, ]

  nb_model <- naiveBayes(Exited ~ ., data = train_data)

  predicted <- predict(nb_model, newdata = test_data)

  correct_predictions <- sum(predicted == test_data$Exited)
  fold_accuracy <- correct_predictions / nrow(test_data)

  accuracy_values[i] <- fold_accuracy
}

cv_accuracy <- mean(accuracy_values)
print(cv_accuracy)

nb_model_full <- naiveBayes(Exited ~ ., data = sample_data3)

predicted_full <- predict(nb_model_full, newdata = sample_data3)

predicted_full <- factor(predicted_full, levels = c("0", "1")) 
sample_data3$Exited <- factor(sample_data3$Exited, levels = c("0", "1"))  

conf_matrix <- confusionMatrix(predicted_full, sample_data3$Exited)
print("Confusion Matrix:")
print(conf_matrix$table)

recall <- conf_matrix$byClass["Recall"]
precision <- conf_matrix$byClass["Precision"]
f_measure <- conf_matrix$byClass["F1"]
print(paste("Recall:", recall))
print(paste("Precision:", precision))
print(paste("F-measure:", f_measure))


