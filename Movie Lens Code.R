# Movielens Project

# Load libraries
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
library(tidyverse)
library(caret)
library(stringr)
library(readr)
library(dplyr)
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

# Clean Data

# 

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

# # Movie Lens Project: Generate movie ratings from the dataset

# Conduct EDA
# Since we conducted some EDA already in our quiz, we will keep this simple
# to focus on modeling:

# Observe distribution of number of ratings by movieId
ggplot(edx %>% count(movieId), aes(x = n)) +
  geom_density(fill = "skyblue", color = "navy", alpha = 0.7) +
  scale_x_log10() +
  labs(title = "Distribution of Ratings per Movie",
       x = "Number of Ratings (log scale)",
       y = "Density") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# Observe distribution of number of ratings by movieId and userId

ggplot(edx %>% count(movieId), aes(x = n)) +
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

# one hot encode the data to get individual genres
genres <- cSplit(edx, "genres", sep = "|", direction = "long")

# Exclude "(no genres listed)" category
genres <- filter(genres, genres != "(no genres listed)")

# Get the top 10 genres
top_genres <- names(sort(table(genres$genres), decreasing = TRUE)[1:10])

# Filter the dataset to include only the top 10 genres
genres_top10 <- filter(genres, genres %in% top_genres)

# Create a bar plot
barplot(table(genres_top10$genres), col = rainbow(10),
        main = "Top 10 Genres Distribution", xlab = "Genres", ylab = "Number of Ratings",
        las = 2, cex.names = 0.8) 


# Generate Machine Learning Models

# Split the edx data into training and test sets
train_index <- createDataPartition(y = edx$rating, times = 1, p = 0.8, list = FALSE)
train_set <- edx[train_index,]
test_set <- edx[-train_index,]

# Set up RMSE function
RMSE <- function(true_ratings, predicted_ratings) {
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Generate a baseline model:
overall_mean_rating <- mean(train_set$rating)
overall_mean_rating

simple_model_rmse <- RMSE(test_set$rating, overall_mean_rating)
simple_model_rmse

# Save results in a table to compare to models later

rmse_summary <- tibble(Model = "Baseline Model", RMSE = simple_model_rmse)
rmse_summary

# Add in the movie-specific effect using average by movie ID, find the RMSE, and save to table:

# Calculate movie effect based on the training set
movie_effect_train <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_m = mean(rating - overall_mean_rating, na.rm = TRUE))

# Left join the movie effect to the test set
combined_table_test <- test_set %>% 
  left_join(movie_effect_train, by = "movieId") 

# Make predictions on the test set, dealing with missing values that may arise
# from joining tables above where movieId values in the test set are not present in the train_set
predicted_ratings_test <- combined_table_test %>%
  mutate(predicted_value = ifelse(is.na(b_m), overall_mean_rating, overall_mean_rating + b_m)) %>%
  pull(predicted_value)

# Calculate RMSE for the test set
model_2_rmse <- RMSE(test_set$rating, predicted_ratings_test)
model_2_rmse

# Update rmse_summary
rmse_summary <- bind_rows(rmse_summary,
                          tibble(Model = "Model 2: Movie effect model",
                                 RMSE = model_2_rmse))
rmse_summary


# Add in the user-specific effect:

movie_and_user_effect <- train_set %>% # create a new table, adding movie-specific effect b_m to train_set table, and group by user ID
  left_join(movie_effect_train, by = "movieId") %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating - overall_mean_rating - b_m)) # user effect = mean(train_set rating - baseline - movie effect)
movie_and_user_effect # creates a table of just userId and user-specific effect

combined_table <- test_set %>% 
  left_join(movie_effect, by = "movieId") %>% 
  left_join(movie_and_user_effect, by = "userId")
combined_table

predicted_ratings <- combined_table %>% mutate(predicted_value = overall_mean_rating + b_m + b_u) %>% 
  pull(predicted_value)

model_3_rmse <- RMSE(predicted_ratings, test_set$rating)
model_3_rmse

rmse_summary <- bind_rows(rmse_summary,
                          tibble(Model = "Model 3: Movie and user effect model",
                                 RMSE = model_3_rmse))



# add in genre-specific effect

genre_movie_user_effect <- train_set %>% # create a new table, adding genre-specific effect b_g to train_set table, and group by genre
  left_join(movie_effect, by = "userId") %>% 
  group_by(genres) %>% 
  summarize(b_g = mean(rating - overall_mean_rating - b_m - b_u)) # genre effect = mean(train_set rating - baseline - movie effect - user effect)

genre_movie_user_effect # creates a table of just userId and genre-specific effect

combined_table <- test_set %>% 
  left_join(movie_effect, by = "movieId") %>% 
  left_join(movie_effect, by = "userId")
combined_table

predicted_ratings <- combined_table %>% mutate(predicted_value = overall_mean_rating + b_m + b_u) %>% 
  pull(predicted_value)

model_4_rmse <- RMSE(predicted_ratings, test_set$rating)
model_4_rmse

rmse_summary <- bind_rows(rmse_summary,
                          tibble(Model = "Model 4: Genre, movie, and user effect model",
                                 RMSE = model_4_rmse))
## One hot encoding

moviegenres_test <- head(train_set %>% select(movieId, genres))
moviegenres_split <- transpose(tstrsplit(moviegenres_test$genres, "|", fixed = TRUE, fill = "NA"))
setNames(do.call(rbind.data.frame, moviegenres_split), c("1", "2", "3", "4"))

moviegenres_split <- as.data.frame(do.call(rbind, moviegenres_split))

moviegenres_movieid <- mutate(moviegenres_split, movieId = head(edx$movieId), .before = V1)

movies_long <- moviegenres_movieid %>% 
  pivot_longer(
    cols = `V1`:`V4`, 
    names_to = "column_name",
    values_to = "genre"
  )


movies_long_coded <- movies_long %>% select(genre)
movies_long_coded <- as.data.frame(model.matrix( ~ . -1, movies_long_coded)) %>% mutate(movieId = movies_long$movieId, .before = genreAction)
movies_long_coded <- movies_long_coded %>% filter(genreNA == 0) %>% select(-(genreNA))

train_set_short <- train_set
test_run <- inner_join(train_set_short, movies_long_coded, by = "movieId")
head(test_run)
