from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import KNNBasic
from surprise import NMF
from surprise import accuracy
import pandas as pd

# Load the MovieLens dataset into a Surprise Dataset object
data = Dataset.load_builtin('ml-1m')

# Define the Reader object
reader = Reader(rating_scale=(1, 5))

# Optionally, convert the Surprise Dataset object into a pandas DataFrame
df = pd.DataFrame(data.raw_ratings, columns=['user_id', 'item_id', 'rating', 'timestamp'])

# Drop the 'timestamp' column if it's not needed
df = df.drop(columns=['timestamp'])

# Load the data into Surprise format using load_from_df
data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)

# Split the data into test and train sets
trainset, testset = train_test_split(data, test_size=0.2)

# Build the KNN model
sim_options = {'name': 'pearson', 'user_based': True}
model_KNN = KNNBasic(sim_options=sim_options)

# Train the model
model_KNN.fit(trainset)

# Build the NMF model
model_NMF = NMF()

# Train the model
model_NMF.fit(trainset)


# Make predictions on the test set
predictions = model_KNN.test(testset)

# Evaluate the model
accuracy.rmse(predictions)

# Make predictions on the test set with the NMF model
predictions = model_NMF.test(testset)

# Evaluate the model
accuracy.rmse(predictions)