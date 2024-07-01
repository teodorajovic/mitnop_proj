# %% Biblioteke
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Bidirectional
from sklearn.utils import resample
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
from collections import Counter


# %% random seed
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

# %% ucitavanje i uredjivanje podatka
filmovi = pd.read_csv("movies.csv")
filmovi = filmovi[["title", "genres"]]
filmovi.dropna(inplace=True)
filmovi = filmovi[filmovi['genres'] != '(no genres listed)']
filmovi['title'] = filmovi['title'].str.replace('"', '')

print(filmovi.head())
filmovi = filmovi.head(1500) 
filmovi['genres'] = filmovi['genres'].apply(lambda x: x.split('|'))

ml = MultiLabelBinarizer()
filmovi = pd.concat([filmovi, pd.DataFrame(ml.fit_transform(filmovi.pop('genres')),
                                         columns=ml.classes_,
                                         index=filmovi.index)], axis=1)

print(filmovi.head())




podaci = pd.read_csv("IMDB.csv", encoding='latin1')


skup = podaci[["Rating","TotalVotes","Genre1","Genre2","Genre3","Budget","Runtime"]]


skup = skup.rename(columns={"Rating":"ocena",
                            "TotalVotes":"broj_glasova",
                            "Budget":"budzet",
                            "Runtime":"vreme"})


skup["zanr"] = skup[["Genre1", "Genre2", "Genre3"]].fillna('').astype(str).agg('|'.join, axis=1)
skup["zanr"] = skup["zanr"].str.strip('|')
skup = skup[skup["zanr"] != '']
skup = skup.drop(columns=["Genre1", "Genre2", "Genre3"])

oznaka_preslikavanja = {"Biography":0,
                        "Drama":1,
                        "History":2,
                        "Adventure":3,
                        "Comedy":4,
                        "Fantasy":5,
                        "Romance":6,
                        "Action":7,
                        "Mystery":8,
                        "Sci-Fi":9,
                        "War":10,
                        "Animation":11,
                        "Thriller":12,
                        "Western":14,
                        "Crime":15,
                        "Family":16,
                        "Music":17,
                        "Musical":18,
                        "Sport":19,
                        "Horror":20}


def map_zanrovi(zanr):
    genres = zanr.split('|')
    mapped_genres = [oznaka_preslikavanja[genre] for genre in genres if genre in oznaka_preslikavanja]
    return mapped_genres

skup['oznaka'] = skup['zanr'].apply(map_zanrovi)


skup['ocena'] = skup['ocena'].astype(float)

def map_ocene(ocena):
    if ocena >= 8.0:
        return "odlican"
    elif ocena >= 7.0:
        return "dobar"
    elif ocena >= 6.0:
        return "prosecan"
    else:
        return "los"

skup['ocena_kategorija'] = skup['ocena'].apply(map_ocene)



skup['broj_glasova'] = skup['broj_glasova'].str.replace(',', '').astype(int)



skup['budzet'] = skup['budzet'].replace('[\$,]', '', regex=True)  
skup['budzet'] = skup['budzet'].str.extract('(\d+)').astype(float)


skup['vreme'] = skup['vreme'].str.extract('(\d+)')
skup = skup.dropna(subset=['vreme'])
skup['vreme'] = skup['vreme'].astype(int)

def map_vreme(vreme):
    if vreme <= 90:
        return "kratak"
    elif vreme <= 120:
        return "srednji"
    else:
        return "dugacak"

skup['vreme_kategorija'] = skup['vreme'].apply(map_vreme)


oznaka_preslikavanja_obrnuto = {v: k for k, v in oznaka_preslikavanja.items()}

def decode_zanrovi(predikcija_oznaka):
    decoded_genres = []
    for genres in predikcija_oznaka:
        decoded = [oznaka_preslikavanja_obrnuto[genre] for genre in genres if genre in oznaka_preslikavanja_obrnuto]
        decoded_genres.append(decoded)
    return decoded_genres

print(skup)

skup.shape
skup.dtypes

skup = skup.reset_index(drop=True)

# %% balansiranje podataka
#balansiranje S
zanrovi = [genre for sublist in skup['oznaka'] for genre in sublist]
zanrovi_brojanje = Counter(zanrovi)
print(zanrovi_brojanje)

max_zanr_count = max(zanrovi_brojanje.values())

balanced_data = []
for genre, count in zanrovi_brojanje.items():
    samples = skup[skup['oznaka'].apply(lambda x: genre in x)]
    if count < max_zanr_count:
        samples = resample(samples, replace=True, n_samples=max_zanr_count, random_state=seed)
    balanced_data.append(samples)

balanced_skup = pd.concat(balanced_data).reset_index(drop=True)

zanrovi_balansirani = [genre for sublist in balanced_skup['oznaka'] for genre in sublist]
zanrovi_brojanje_balansirani = Counter(zanrovi_balansirani)
print("Distribucija žanrova nakon balansiranja:")
print(zanrovi_brojanje_balansirani)


#balansiranje T
zanrovi_filmova = [genre for sublist in filmovi.drop(columns=['title']).values for genre in sublist if genre == 1]
br_zanrova_filmova = Counter(zanrovi_filmova)
print(br_zanrova_filmova)

max_br_zanrova_filmova = max(br_zanrova_filmova.values())

balansirani_podaci = []
for genre in ml.classes_:
    samples = filmovi[filmovi[genre] == 1]
    count = len(samples)
    if count < max_br_zanrova_filmova:
        samples = resample(samples, replace=True, n_samples=max_br_zanrova_filmova, random_state=seed)
    balansirani_podaci.append(samples)

balansirani_filmovi = pd.concat(balansirani_podaci).reset_index(drop=True)

balanced_genres_movies = [genre for sublist in balansirani_filmovi.drop(columns=['title']).values for genre in sublist if genre == 1]
balanced_genres_count_movies = Counter(balanced_genres_movies)
print("Distribucija žanrova nakon balansiranja:")
print(balanced_genres_count_movies)


# %% grafikoni distributivnosti zanrova
plt.figure(figsize=(12, 6))
plt.bar(zanrovi_brojanje.keys(), zanrovi_brojanje.values())
plt.title('Distribucija žanrova pre balansiranja')
plt.xlabel('Žanrovi')
plt.ylabel('Broj uzoraka')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
plt.bar(zanrovi_brojanje_balansirani.keys(), zanrovi_brojanje_balansirani.values())
plt.title('Distribucija žanrova nakon balansiranja')
plt.xlabel('Žanrovi')
plt.ylabel('Broj uzoraka')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



br_zanr = filmovi.drop(columns=['title']).sum()

plt.figure(figsize=(10, 8))
br_zanr.plot(kind='bar')
plt.title('Raspodela žanrova pre balansiranja')
plt.xlabel('Žanr')
plt.ylabel('Broj filmova')
plt.show()


genre_counts_balanced_movies = balansirani_filmovi.drop(columns=['title']).sum()

plt.figure(figsize=(10, 8))
genre_counts_balanced_movies.plot(kind='bar')
plt.title('Raspodela žanrova nakon balansiranja')
plt.xlabel('Žanr')
plt.ylabel('Broj filmova')
plt.show()

# %% tokenizacija
tokenizer = Tokenizer(num_words=5000, lower=True, split=' ')
tokenizer.fit_on_texts(balansirani_filmovi['title'].values)
x = tokenizer.texts_to_sequences(balansirani_filmovi['title'].values)
x = pad_sequences(x)

# %% Priprema podataka
X = balanced_skup[['ocena', 'broj_glasova', 'budzet', 'vreme']].values
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(balanced_skup['oznaka'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



y = balansirani_filmovi.drop(columns=['title']).values

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=seed)

print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)


# %% obucavanje
#ANN trening sa numerickim podacima
nm = Sequential()
nm.add(Dense(256, input_dim=X_train.shape[1], activation='relu'))
nm.add(Dense(128, activation='relu'))
nm.add(Dense(64, activation='relu'))
nm.add(Dense(32, activation='relu'))
nm.add(Dense(y_train.shape[1], activation='sigmoid'))

nm.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

hist = nm.fit(X_train, Y_train, epochs=100, batch_size=16, validation_data=(X_test, Y_test))

loss, accuracy = nm.evaluate(X_test, Y_test)



#RNN trening sa tekstualnim podacima
rnn = Sequential()
rnn.add(Embedding(input_dim=5000, output_dim=64))  
rnn.add(SpatialDropout1D(0.5))
rnn.add(Bidirectional(LSTM(units=64, dropout=0.5, recurrent_dropout=0.5)))
rnn.add(Dense(units=y.shape[1], activation='tanh'))

rnn.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

rnn.build(input_shape=(None, x.shape[1]))
rnn.summary()

zaustavljanje = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

history = rnn.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_val, y_val), callbacks=[zaustavljanje])


# %% evaulacija
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(hist.history['loss'], label='Training Loss')
plt.plot(hist.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(hist.history['accuracy'], label='Training Accuracy')
plt.plot(hist.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


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


# %% predikcija
naslov_filma = "vampire"
seq = tokenizer.texts_to_sequences([naslov_filma])
padded = pad_sequences(seq, maxlen=x.shape[1])
pred = rnn.predict(padded,verbose=0)
top = pred[0].argsort()[-3:][::-1]
pred_zanra = [ml.classes_[i] for i in top if pred[0][i] > 0.2]
print(f"Za naslov '{naslov_filma}', predviđeni žanr(i) su: {pred_zanra}")


# %% koriscenje

#print(mlb.classes_)
#print(movies.head())

example_titles = ["romantic",
                  "adventure time",
                   "money", 
                   "mafia", 
                   "space", 
                   "war documentary", 
                   "Toy Story", 
                   "blood", 
                   "love", 
                   "relationship"]

for title in example_titles:
    seq = tokenizer.texts_to_sequences([title])
    padded = pad_sequences(seq, maxlen=x.shape[1])
    pred = rnn.predict(padded,verbose=0)
    top = pred[0].argsort()[-3:][::-1]
    pred_zanra = [ml.classes_[i] for i in top if pred[0][i] > 0.2]
    
    print(f"Za naslov '{title}', predviđeni žanr(i) su: {pred_zanra}\n")

# %% validacija
predikcije = nm.predict(X_test)
predikcije = (predikcije > 0.5).astype(int)
predikcija_oznaka = mlb.inverse_transform(predikcije)


# %% koriscenje
examples = np.array([
    [8.2, 15000, 12000000, 135],
    [6.5, 8000, 2500000, 95],
    [7.1, 12000, 7500000, 110],
    [8.5, 11000, 15000000, 145],
    [5.9, 5000, 2000000, 85],
    [7.3, 16000, 8000000, 105],   
    [8.0, 18000, 10000000, 130],  
    [6.8, 9000, 3000000, 90],    
    [7.5, 10000, 8500000, 115]  
])

examples_scaled = scaler.transform(examples)

# Predikcija
example_predictions = nm.predict(examples_scaled)
example_predictions = (example_predictions > 0.5).astype(int)
example_genres = mlb.inverse_transform(example_predictions)

examples_original = scaler.inverse_transform(examples_scaled)

for i, (ocena, broj_glasova, budzet, vreme) in enumerate(examples_original):
    genres = decode_zanrovi([example_genres[i]])
    print(f"Primer {i+1}: Za ocenu {ocena:.2f}, broj glasova {broj_glasova:.0f}, budžet {budzet:.0f} i vreme trajanja {vreme:.0f},\n predviđeni žanrovi su {genres}\n")