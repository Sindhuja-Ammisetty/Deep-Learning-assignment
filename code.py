import numpy as np
import pandas as pd
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Load the IMDB dataset
max_features = 20000  # Number of unique words to consider
maxlen = 100  # Cut texts after this number of words
batch_size = 64

print("Loading data...")
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

# Pad sequences to ensure uniform input size
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

# Build the LSTM model
model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
print("Training model...")
history = model.fit(X_train, y_train, epochs=5, batch_size=batch_size, validation_data=(X_test, y_test), verbose=2)

# Evaluate the model
print("Evaluating model...")
score, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=2)
print(f'Test score: {score}, Test accuracy: {accuracy}')

# Example of predicting a new review
def predict_review(review):
    # Preprocess the review
    review_seq = imdb.get_word_index()
    review_seq = {v: k for k, v in review_seq.items()}
    review_words = review.lower().split()
    review_indices = [review_seq[word] for word in review_words if word in review_seq]
    review_padded = pad_sequences([review_indices], maxlen=maxlen)
    
    # Predict
    prediction = model.predict(review_padded)
    return "Positive" if prediction[0][0] > 0.5 else "Negative"

# Test the prediction function
new_review = "I loved this movie! It was fantastic."
print(f"Review: {new_review} - Prediction: {predict_review(new_review)}")
