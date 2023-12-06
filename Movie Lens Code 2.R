# HarvardX: PH125.8x
# Data Science: Machine Learning
# R code from course videos

# Model Fitting and Recommendation Systems

# Movielens dataset

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
library(tidyverse)
library(caret)
library(stringr)
library(readr)
library(dplyr)
library(dslabs)
library(data.table)

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)

colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")

movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")  

# Final hold-out test set will be 10% of MovieLens data
# NOTE: The hold-out test is NOT to be used throughout this code
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later

test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Split edx dataset into test and train sets

test_index <- createDataPartition(y = edx$rating, times = 1,
                                  p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Define RMSE function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Run basic movie effect model
mu_hat <- mean(train_set$rating)
naive_rmse <- RMSE(test_set$rating, mu_hat)
rmse_results <- data_frame(method = "Just the average", RMSE = naive_rmse)
mu <- mean(train_set$rating) 
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))
predicted_ratings <- mu + test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)
model_1_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie Effect Model",
                                     RMSE = model_1_rmse ))

# user effect model
user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
model_2_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie + User Effects Model",  
                                     RMSE = model_2_rmse ))
# genre effect model
genre_avgs <- train_set %>%
  left_join(movie_avgs, by="movieId") %>% 
  left_join(user_avgs, by="userId") %>% 
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u))
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by="genres") %>% 
  mutate(pred = mu + b_i + b_u + b_g) %>%
  pull(pred)
model_3_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie + User + Genre Effects Model",  
                                     RMSE = model_3_rmse ))


# Apply regularization to the movie + user + genre model
# WARNING: This code will take at least 5 mins to run depending on machine capabilities

# Regularization parameter search for Movie + User + Genre Effect Model
lambdas <- seq(0, 10, 0.25)

# Initialize an empty dataframe to store results
rmse_results_lambda <- data.frame(Lambda = numeric(), RMSE = numeric())

for (lambda in lambdas) {
  # Movie effect model with regularization
  movie_avgs_reg <- train_set %>% 
    group_by(movieId) %>% 
    summarize(b_i = sum(rating - mu) / (n() + lambda))
  
  # User effect model with regularization
  user_avgs_reg <- train_set %>% 
    left_join(movie_avgs_reg, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_i) / (n() + lambda))
  
  # Genre effect model with regularization
  genre_avgs_reg <- train_set %>%
    left_join(movie_avgs_reg, by = "movieId") %>% 
    left_join(user_avgs_reg, by = "userId") %>% 
    group_by(genres) %>%
    summarize(b_g = sum(rating - mu - b_i - b_u) / (n() + lambda))
  
  # Predict on the test set
  predicted_ratings <- test_set %>% 
    left_join(movie_avgs_reg, by = "movieId") %>%
    left_join(user_avgs_reg, by = "userId") %>%
    left_join(genre_avgs_reg, by = "genres") %>% 
    mutate(pred = mu + b_i + b_u + b_g) %>%
    pull(pred)
  
  # Calculate RMSE
  model_rmse <- RMSE(predicted_ratings, test_set$rating)
  
  # Store the results
  rmse_results_lambda <- bind_rows(rmse_results_lambda,
                                   data.frame(Lambda = lambda, RMSE = model_rmse))
}

# Plot RMSE vs. lambda
qplot(Lambda, RMSE, data = rmse_results_lambda) + geom_point() +
  labs(title = "RMSE vs. Lambda for Regularized Movie + User + Genre Effect Model")

# Find the lambda with the minimum RMSE
lambda_min <- rmse_results_lambda$Lambda[which.min(rmse_results_lambda$RMSE)]

# Display the optimal lambda
cat("Optimal Lambda:", lambda_min, "\n")

# Update rmse_results
rmse_results <- bind_rows(rmse_results,
                          data_frame(method = "Regularized Movie + User + Genre Effect Model",  
                                     RMSE = min(rmse_results_lambda$RMSE)))
rmse_results
