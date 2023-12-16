from keras.datasets import imdb
from keras import models
from keras import layers
from keras import optimizers

# Veri yükleme
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# Örnek verileri gösterme
print(train_data[0])

# Word index oluşturma
word_index = imdb.get_word_index()

# Tersine çevrilmiş word index oluşturma
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# Veriyi metne dönüştürme
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

# Veriyi vektörleştirme
import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# Etiketleri vektörleştirme
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# Model oluşturma
model = models.Sequential()
model.add(layers.Dense(16, activation="relu", input_shape=(10000,)))
model.add(layers.Dense(16, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

# Modeli derleme
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])

# Modeli eğitme
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

# Eğitim geçmişi
history_dict = history.history
print(history_dict.keys())

#%%
import matplotlib.pyplot as plt 

history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]

epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'bo', label="Eğitim Kaybı")
plt.plot(epochs, val_loss_values, 'b', label="Doğruluk Kaybı")

plt.title("Eğitim ve Doğruluk Kaybı")
plt.xlabel("Epoklar")
plt.ylabel("Kayıp")
plt.legend()

plt.show()

#%%
plt.clf()

acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]

plt.plot(epochs, loss_values, 'bo', label="Eğitim Kaybı")
plt.plot(epochs, val_loss_values, 'b', label="Doğruluk Kaybı")

plt.title("Eğitim ve Doğruluk Kaybı")
plt.xlabel("Epoklar")
plt.ylabel("Kayıp")
plt.legend()

plt.show()

#%%

model=models.Sequential()
model.add(layers.Dense(16, activation="relu", input_shape=(10000,)))
model.add(layers.Dense(16, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])

model.fit(x_train , y_train , epochs = 4, batch_size = 512)
results = model.evaluate(x_test , y_test)

results