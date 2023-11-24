# Importações necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical

# Carregando os dados de treinamento e validação
train_data = pd.read_csv('trainReduzido.csv', index_col=0)
validation_data = pd.read_csv('validacao.csv', index_col=0)

# Separando rótulos (dígitos) e imagens do conjunto de treinamento
train_labels = train_data['label']
train_images = train_data.drop('label', axis=1)

# Normalizando os pixels das imagens para valores entre 0 e 1
train_images = train_images / 255.0
validation_images = validation_data / 255.0

# Convertendo rótulos para representação one-hot encoding
train_labels_one_hot = to_categorical(train_labels, num_classes=10)

# Redimensionando os dados de imagem para o formato adequado para o modelo de rede neural
train_images = train_images.values.reshape(-1, 28, 28, 1)
validation_images = validation_images.values.reshape(-1, 28, 28, 1)

# Dividindo os dados de treinamento em subconjuntos de treinamento e teste
train_images_train, train_images_test, train_labels_train, train_labels_test = train_test_split(
    train_images, train_labels_one_hot, test_size=0.2, random_state=42
)

# Construindo o modelo de rede neural
model = Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(700, activation='tanh'),
    Dropout(0.0),
    Dense(10, activation='sigmoid')
])

# Compilando o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinando o modelo com os dados de treinamento
history = model.fit(train_images_train, train_labels_train, epochs=10, batch_size=128, validation_split=0.2)

# Visualizando o histórico de acurácia e perda durante o treinamento
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

# Fazendo previsões no subconjunto de teste
test_predictions = model.predict(train_images_test)
test_predicted_labels = np.argmax(test_predictions, axis=1)
test_true_labels = np.argmax(train_labels_test, axis=1)

# Identificando os índices dos acertos e erros
corrects = np.where(test_predicted_labels == test_true_labels)[0]
incorrects = np.where(test_predicted_labels != test_true_labels)[0]

# Selecionando aleatoriamente exemplos de acertos e erros
selected_corrects = np.random.choice(corrects, 5, replace=False)
selected_incorrects = np.random.choice(incorrects, 5, replace=False)

# Plotando exemplos de acertos
plt.figure(figsize=(10, 5))
for i, correct in enumerate(selected_corrects):
    plt.subplot(1, 5, i + 1)
    plt.imshow(train_images_test[correct].reshape(28, 28), cmap='gray')
    plt.title(f"Correto: {test_true_labels[correct]}")
    plt.axis('off')
plt.suptitle("Exemplos de Acertos")
plt.show()

# Plotando exemplos de erros
plt.figure(figsize=(10, 5))
for i, incorrect in enumerate(selected_incorrects):
    plt.subplot(1, 5, i + 1)
    plt.imshow(train_images_test[incorrect].reshape(28, 28), cmap='gray')
    plt.title(f"Errado: {test_predicted_labels[incorrect]}")
    plt.axis('off')
plt.suptitle("Exemplos de Erros")
plt.show()

# Fazendo previsões no conjunto de validação para submissão
validation_predictions = model.predict(validation_images)
validation_predicted_labels = np.argmax(validation_predictions, axis=1)

# Preparando o arquivo de submissão com as previsões
submission = pd.DataFrame({
    'ImageId': range(1, len(validation_predicted_labels) + 1),
    'Label': validation_predicted_labels
})

# Salvando o arquivo de submissão
submission.to_csv('submission.csv', index=False)
