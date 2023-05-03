from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

# завантаження даних
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# попередня обробка даних
# розгортаємо зображення у вектор розміром 784 (28х28)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
# конвертуємо тип даних у float32 та нормалізуємо до діапазону [0, 1]
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
# конвертуємо мітки класів у бінарний вектор
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# побудова моделі
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# компіляція моделі
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# навчання моделі
model.fit(X_train, y_train, batch_size=128, epochs=10, verbose=1, validation_data=(X_test, y_test))

# оцінка точності моделі на тестових даних
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])