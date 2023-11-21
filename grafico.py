import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical

# Carregar os dados
train_data = pd.read_csv('trainReduzido.csv', index_col=0)
validation_data = pd.read_csv('validacao.csv', index_col=0)

# Separar os rótulos e os pixels
train_labels = train_data['label']
train_images = train_data.drop('label', axis=1)

# Verificar as classes nos rótulos
unique_classes = np.unique(train_labels)
print("Classes únicas nos rótulos de treinamento:", unique_classes)

# Normalizar os pixels
train_images = train_images / 255.0
validation_images = validation_data / 255.0

# Converter os rótulos para one-hot encoding
# Assegurar que o número de classes é 10
train_labels = to_categorical(train_labels, num_classes=10)

# Redimensionar os dados de entrada para o modelo
train_images = train_images.values.reshape(-1, 28, 28)
validation_images = validation_images.values.reshape(-1, 28, 28)

# Visualizar algumas imagens de dígitos
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(np.argmax(train_labels[i]))
plt.show()

# Definir o modelo
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(512, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

# Compilar o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo e guardar o histórico
history = model.fit(train_images, train_labels, epochs=10, batch_size=128, validation_split=0.2)

# Visualizar o histórico de treinamento
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Acurácia no Treinamento')
plt.plot(history.history['val_accuracy'], label='Acurácia na Validação')
plt.title('Acurácia ao longo das Épocas')
plt.ylabel('Acurácia')
plt.xlabel('Época')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Perda no Treinamento')
plt.plot(history.history['val_loss'], label='Perda na Validação')
plt.title('Perda ao longo das Épocas')
plt.ylabel('Perda')
plt.xlabel('Época')
plt.legend()
plt.show()

# Fazer previsões
predictions = model.predict(validation_images.reshape(-1, 28, 28))
predicted_labels = np.argmax(predictions, axis=1)

# Preparar o arquivo de submissão
submission = pd.DataFrame({
    'ImageId': range(1, len(predicted_labels) + 1),
    'Label': predicted_labels
})
submission.to_csv('submission.csv', index=False)
