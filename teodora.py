# %% lib

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# %% load data

movies = pd.read_csv("movies.csv")
movies = movies[["title", "genres"]]
movies.dropna(inplace=True)
movies = movies[movies['genres'] != '(no genres listed)']
movies['title'] = movies['title'].str.replace('"', '')

print(movies.head())

# movies = movies.head(500)  # Uncomment to use a subset of data

movies['genres'] = movies['genres'].apply(lambda x: x.split('|'))

mlb = MultiLabelBinarizer()
movies = pd.concat([movies, pd.DataFrame(mlb.fit_transform(movies.pop('genres')),
                                         columns=mlb.classes_,
                                         index=movies.index)], axis=1)

print(movies.head())

# Plot genre distribution to check for class imbalance
genre_counts = movies.drop(columns=['title']).sum()

plt.figure(figsize=(10, 8))
genre_counts.plot(kind='bar')
plt.title('Raspodela žanrova')
plt.xlabel('Žanr')
plt.ylabel('Broj filmova')
plt.show()

# %% title tokenization

tokenizer = Tokenizer(num_words=5000, lower=True, split=' ')
tokenizer.fit_on_texts(movies['title'].values)
X = tokenizer.texts_to_sequences(movies['title'].values)
X = pad_sequences(X)

# %% prep and training
Y = movies.drop(columns=['title']).values

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

print(X_train.shape, X_val.shape, Y_train.shape, Y_val.shape)

# %% RNN
rnn = Sequential()
rnn.add(Embedding(input_dim=5000, output_dim=64))  # Increased output dimension
rnn.add(SpatialDropout1D(0.3))
rnn.add(Bidirectional(LSTM(units=64, dropout=0.3, recurrent_dropout=0.3)))  # Added Bidirectional LSTM
rnn.add(Dense(units=Y.shape[1], activation='tanh'))

rnn.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

rnn.build(input_shape=(None, X.shape[1]))
rnn.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = rnn.fit(X_train, Y_train, epochs=20, batch_size=256, validation_data=(X_val, Y_val), callbacks=[early_stopping], verbose=2)

# %% evaluation 

plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Loss during training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('Accuracy during training')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# %% prediction

def predikcija_zanra(naslov):
    seq = tokenizer.texts_to_sequences([naslov])
    padded = pad_sequences(seq, maxlen=X.shape[1])
    pred = rnn.predict(padded)
    top_indices = pred[0].argsort()[-3:][::-1]
    pred_genres = [mlb.classes_[i] for i in top_indices if pred[0][i] > 0.2]
    return pred_genres

naslov_filma = "vampire"
predikcija = predikcija_zanra(naslov_filma)
print(f"Za naslov '{naslov_filma}', predviđeni žanr(i) su: {predikcija}")

# %% additional info

#print(mlb.classes_)
#print(movies.head())

example_titles = ["romantic", "adventure time", "money" , "mafia",
                  "space", "war documentary","Toy Story", "blood","love","relationship"]
for title in example_titles:
    predikcija = predikcija_zanra(title)
    print(f"Za naslov '{title}', predviđeni žanr(i) su: {predikcija}")
