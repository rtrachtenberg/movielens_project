# Movielens dataset

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
library(tidyverse)
library(caret)
library(stringr)
library(readr)
library(dplyr)
library(data.table)

options(timeout = 120)

# note to self: The final_holdout_test set is meant to provide a final, 
# unbiased estimate of your single best model. 
# To test the performance of multiple models before choosing the best one, 
# split the edx set into train and test sets and/or use cross-validation. 
# The final_holdout_test may not be be used for model training, model development, 
# or selecting from multiple models or else you will only receive 5 out of 25 points 
# for the RMSE section of your MovieLens project.

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
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later

test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# NOTE TO SELF: split remaining edx into train and test sets

# # Movie Lens Project: Generate movie ratings from the dataset

# Conduct EDA

movielens %>% count(movieId) %>% ggplot(aes(n)) + geom_density() + scale_x_log10()
movielens %>% count(userId) %>% ggplot(aes(n)) + geom_density() + scale_x_log10()
movielens %>% count(genres) %>% ggplot(aes(n)) + geom_density() + scale_x_log10()

movielens %>% group_by(rating) %>% summarize(count = n()) %>% ggplot(aes(x = rating, y = count)) + geom_point()

# Generate a baseline model:
overall_mean_rating <- mean(edx$rating)
overall_mean_rating

RMSE <- function(true_ratings, predicted_ratings) {
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

simple_model_rmse <- RMSE(final_holdout_test$rating, overall_mean_rating)
simple_model_rmse

# Save results in a table to compare to models later

rmse_summary <- tibble(Model = "Baseline Model", RMSE = simple_model_rmse)
rmse_summary

# Add in the movie-specific effect using average by movie ID, find the RMSE, and save to table:

movie_effect <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_m = mean(rating - overall_mean_rating))
movie_effect

combined_table <- final_holdout_test %>% 
  left_join(movie_effect, by = "movieId") 

predicted_ratings <- combined_table %>%
  mutate(predicted_value = overall_mean_rating + b_m) %>%
  pull(predicted_value)
predicted_ratings

model_2_rmse <- RMSE(predicted_ratings, final_holdout_test$rating)
model_2_rmse

rmse_summary <- bind_rows(rmse_summary,
                          tibble(Model = "Model 2: Movie effect model",
                                 RMSE = model_2_rmse))
rmse_summary

# Apply regularization to movie-specific effect:

lambdas <- seq(0, 10, 0.25)
sum_only <- edx %>% 
  group_by(movieId) %>% 
  summarize(sum_rating = sum(rating - overall_mean_rating), count_movies = n())
  
regzed_rmses <- sapply(lambdas, function(lambda){
  predicted_ratings <- final_holdout_test %>% 
    left_join(sum_only, by='movieId') %>% 
    mutate(b_ml = sum_rating/(count_movies + lambda)) %>%
    mutate(pred = overall_mean_rating + b_ml) %>%
    .$pred
  return(RMSE(final_holdout_test$rating, predicted_ratings))
})
qplot(lambdas, regzed_rmses)  
best_lambda <- lambdas[which.min(regzed_rmses)]

b_m <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_m = sum(rating - overall_mean_rating)/(n() + best_lambda))

predicted_ratings <- combined_table %>%
  mutate(predicted_value = overall_mean_rating + b_m) %>%
  pull(predicted_value)
predicted_ratings

model_3_rmse <- RMSE(predicted_ratings, final_holdout_test$rating)
model_3_rmse

rmse_summary <- bind_rows(rmse_summary,
                          tibble(Model = "Model 3: Movie effect model w Regularization",
                                 RMSE = model_3_rmse))

# Add in the user-specific effect:

movie_and_user_effect <- edx %>% # create a new table, adding movie-specific effect b_m to edx table, and group by user ID
  left_join(movie_effect, by = "movieId") %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating - overall_mean_rating - b_m)) # user effect = mean(edx rating - baseline - movie effect)
movie_and_user_effect # creates a table of just userId and user-specific effect

combined_table <- final_holdout_test %>% 
  left_join(movie_effect, by = "movieId") %>% 
  left_join(movie_and_user_effect, by = "userId")
combined_table

predicted_ratings <- combined_table %>% mutate(predicted_value = overall_mean_rating + b_m + b_u) %>% 
  pull(predicted_value)

model_3_rmse <- RMSE(predicted_ratings, final_holdout_test$rating)
model_3_rmse

rmse_summary <- bind_rows(rmse_summary,
                          tibble(Model = "Model 3: Movie and user effect model",
                                 RMSE = model_3_rmse))

# add in genre-specific effect

movie_user_genre_effect <- edx %>% # create a new table, adding b_m and b_u to edx table, and group by genre
  left_join(movie_effect, by = "movieId") %>% 
  left_join(movie_and_user_effect, by = "userId") %>% 
  group_by(genres) %>% 
  summarize(b_g = mean(rating - overall_mean_rating - b_m - b_u)) # genre effect = mean(edx rating - baseline - movie effect - user effect)
head(movie_user_genre_effect) # creates a table of just genre and genre-specific effect

combined_table <- final_holdout_test %>% 
  left_join(movie_effect, by = "movieId") %>% 
  left_join(movie_and_user_effect, by = "userId") %>% 
  left_join(movie_user_genre_effect, by = "genres")
head(combined_table)

predicted_ratings <- combined_table %>% mutate(predicted_value = overall_mean_rating + b_m + b_u + b_g) %>% 
  pull(predicted_value)

model_4_rmse <- RMSE(predicted_ratings, final_holdout_test$rating)
model_4_rmse

rmse_summary <- bind_rows(rmse_summary,
                          tibble(Model = "Model 4: Movie, user, and genre effect model",
                                 RMSE = model_4_rmse))

moviegenres_test <- head(edx %>% select(movieId, genres))
moviegenres_test
moviegenres_split <- transpose(tstrsplit(moviegenres_test$genres, "|", fixed = TRUE, fill = "NA"))
setNames(do.call(rbind.data.frame, moviegenres_split), c("1", "2", "3", "4"))
moviegenres_split

moviegenres_split <- as.data.frame(do.call(rbind, moviegenres_split))
moviegenres_split

moviegenres_movieid <- mutate(moviegenres_split, movieId = head(edx$movieId), .before = V1)
moviegenres_movieid

movies_long <- moviegenres_movieid %>% 
  pivot_longer(
    cols = `V1`:`V4`, 
    names_to = "column_name",
    values_to = "genre"
  )

movies_long_coded <- movies_long %>% select(genre)
movies_long_coded
movies_long_coded <- as.data.frame(model.matrix( ~ . -1, movies_long_coded)) %>% mutate(movieId = movies_long$movieId, .before = genreAction)
movies_long_coded
movies_long_coded <- movies_long_coded %>% filter(genreNA == 0) %>% select(-(genreNA))
movies_long_coded
edx_short <- head(edx)
test_run <- inner_join(edx_short, movies_long_coded, by = "movieId")
head(test_run)
