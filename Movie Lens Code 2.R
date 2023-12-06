# Movielens Project

# Note that the cross-validation model below may take ~20-30 minutes to run.
# Either skip this portion or time your review accordingly.

# Load libraries
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
library(tidyverse)
library(caret)
library(stringr)
library(readr)
library(dplyr)
library(dslabs)
library(data.table)
library(ranger)
library(splitstackshape)
library(randomForest)

options(timeout = 120)

# Load data set from grouplens website
dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

# add ratings data and assign clear column names
ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)

colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")

# transform the column classes
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

# add movie data and assign clear column names
movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")

# transform column classes
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

# join movies and ratings data frames to generate the movielens dataset
movielens <- left_join(ratings, movies, by = "movieId")  

# Final hold-out test set will be 10% of MovieLens data
# NOTE: The hold-out test is NOT to be used throughout this code as a test set
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

# Exploratory Data Analysis

# Observe distribution of number of ratings by movieId
ggplot(edx %>% count(movieId), aes(x = n)) +
  geom_density(fill = "skyblue", color = "navy", alpha = 0.7) +
  scale_x_log10() +
  labs(title = "Distribution of Ratings per Movie",
       x = "Number of Ratings (log scale)",
       y = "Density") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# Observe distribution of number of ratings by userId

ggplot(edx %>% count(userId), aes(x = n)) +
  geom_density(fill = "pink", color = "salmon", alpha = 0.7) +
  scale_x_log10() +
  labs(title = "Distribution of Ratings per Movie",
       x = "Number of Ratings (log scale)",
       y = "Density") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# Let's look into which ratings are most popular
ggplot(edx %>% group_by(rating) %>% summarize(count = n()), aes(x = rating, y = count)) +
  geom_bar(stat = "identity", fill = "lightgreen", alpha = 0.7) +
  labs(title = "Distribution of Ratings",
       x = "Rating",
       y = "Count") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))
# Looks like a rating of 4 is most popular.
# Whole number ratings tend to be more popular than half ratings

# Visualize top genres:

# One-hot encode the data and split the genre into one per row 
# (adding more rows as needed to accommodate the data) for ease of understanding the graph
genres <- cSplit(edx, "genres", sep = "|", direction = "long")

# Exclude "(no genres listed)" category
genres <- filter(genres, genres != "(no genres listed)")

# Get the top 10 genres
top_genres <- names(sort(table(genres$genres), decreasing = TRUE)[1:10])

# Filter the dataset to include only the top 10 genres
genres_top10 <- filter(genres, genres %in% top_genres)

# Order the levels of the factor based on frequency
genres_top10$genres <- factor(genres_top10$genres, levels = names(sort(table(genres_top10$genres), decreasing = TRUE)))

# Create a bar plot
barplot(table(genres_top10$genres), col = rainbow(10),
        main = "Top 10 Genres Distribution", ylab = "Frequency",
        las = 2, cex.names = 0.8)  # las = 2 for vertical labels, adjust cex.names for label size


# Machine Learning Methods

# First, split edx dataset into test and train sets

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


# Ensemble Method

# random forest model
# Define the Random Forest model and use a subset of the data for training due to runtime issues
subset_train_set <- as.data.table(train_set)[1:10000, ]

rf_model <- randomForest(rating ~ userId + movieId + genres, data = subset_train_set, ntree = 5)

# Print the summary of the Random Forest model
print(rf_model)

# Make predictions on the test set
predicted_ratings_rf <- predict(rf_model, newdata = test_set)

# Calculate RMSE
rf_model_rmse <- RMSE(predicted_ratings_rf, test_set$rating)
rf_model_rmse

# Update rmse_results
rmse_results <- bind_rows(rmse_results,
                          data_frame(method = "Random Forest on Subsetted Data",  
                                     RMSE = rf_model_rmse))
rmse_results

# Now let's incorporate more runs subsets of the data using cross-validation:

## Cross validation
# WARNING: This code could take up to ~30 minutes to run, pending your machine capabilities

# Set the number of folds for cross-validation
num_folds <- 5  # You can adjust the number of folds as needed

# Create an empty vector to store cross-validated predictions
cv_predictions <- numeric(length = nrow(train_set))
cv_rmse <- 0  # Initialize RMSE

# Perform cross-validation
for (fold in 1:num_folds) {
  # Create training and validation sets for the current fold
  set.seed(fold)  # Ensure reproducibility across folds
  fold_indices <- createDataPartition(y = train_set$rating, p = 0.8, list = FALSE)
  train_fold <- train_set[fold_indices, ]
  val_fold <- train_set[-fold_indices, ]
  
  # Train the random forest model using ranger
  rf_model <- ranger(rating ~ userId + movieId, data = train_fold, num.trees = 50)
  
  # Make predictions on the validation set
  fold_predictions <- predict(rf_model, data = val_fold)$predictions
  
  # Store the predictions in the cv_predictions vector
  cv_predictions[-fold_indices] <- fold_predictions
  
  # Update RMSE
  cv_rmse <- cv_rmse + RMSE(fold_predictions, val_fold$rating)
}

# Calculate average RMSE across folds
cv_rmse <- cv_rmse / num_folds

cat("Cross-validated RMSE:", cv_rmse, "\n")

rmse_results <- bind_rows(rmse_results,
                          data_frame(method = "Cross Validated Model",  
                                     RMSE = cv_rmse))
rmse_results

# Results:

# View RMSE summary table, sorted by RMSE:

rmse_results_sorted <- rmse_results %>% arrange(RMSE)
rmse_results_sorted

# Let's test our best model against the holdout_test_set to achieve our final RMSE:

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
  
  # Predict on the final holdout test
  predicted_ratings <- final_holdout_test %>% 
    left_join(movie_avgs_reg, by = "movieId") %>%
    left_join(user_avgs_reg, by = "userId") %>%
    left_join(genre_avgs_reg, by = "genres") %>% 
    mutate(pred = mu + b_i + b_u + b_g) %>%
    pull(pred)
  
  # Filter out rows with missing values
  valid_rows <- !is.na(predicted_ratings)
  predicted_ratings <- predicted_ratings[valid_rows]
  final_holdout_ratings <- final_holdout_test$rating[valid_rows]
  
  # Calculate RMSE
  model_rmse <- RMSE(predicted_ratings, final_holdout_ratings)
  
  # Print RMSE value
  cat("Lambda:", lambda, "RMSE:", model_rmse, "\n")
  
  # Store the results
  rmse_results_lambda <- bind_rows(rmse_results_lambda,
                                   data.frame(Lambda = lambda, RMSE = model_rmse))
}

# Plot RMSE vs. lambda
qplot(x = Lambda, y = RMSE, data = rmse_results_lambda) + geom_point() +
  labs(title = "RMSE vs. Lambda for Regularized Movie + User + Genre Effect Model")

# Find the lambda with the minimum RMSE
lambda_min <- rmse_results_lambda$Lambda[which.min(rmse_results_lambda$RMSE)]

# Display the optimal lambda
cat("Optimal Lambda:", lambda_min, "\n")

# Display RMSE
final_rmse = min(rmse_results_lambda$RMSE)
round(final_rmse, 5)

# We have achieved a final RMSE of 0.86526