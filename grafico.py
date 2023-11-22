import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical, plot_model

# Carregar os dados
train_data = pd.read_csv('trainReduzido.csv', index_col=0)
validation_data = pd.read_csv('validacao.csv', index_col=0)

# Separar os rótulos e os pixels
train_labels = train_data['label']
train_images = train_data.drop('label', axis=1)

# Normalizar os pixels
train_images = train_images / 255.0
validation_images = validation_data / 255.0

# Converter os rótulos para one-hot encoding
# Assegurar que o número de classes é 10
train_labels_one_hot = to_categorical(train_labels, num_classes=10)

# Redimensionar os dados de entrada para o modelo
train_images = train_images.values.reshape(-1, 28, 28)
validation_images = validation_images.values.reshape(-1, 28, 28)

# Visualizar algumas imagens de dígitos para cada classe presente
examples_per_class = 5

# Identificar as classes presentes nos dados
classes_presentes = np.unique(train_labels)

# Criar subplots para as classes presentes
fig, axes = plt.subplots(len(classes_presentes), examples_per_class, figsize=(10, len(classes_presentes) * 2))
fig.tight_layout(pad=3.0)

for class_idx, digit in enumerate(classes_presentes):
    idxs = np.where(train_labels == digit)[0]
    idxs = np.random.choice(idxs, examples_per_class, replace=False)
    for j, idx in enumerate(idxs):
        ax = axes[class_idx, j]
        ax.imshow(train_images[idx], cmap='gray')
        ax.axis('off')
        if j == 0:
            ax.set_title(f"Classe {digit}")

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

# Sumário do Modelo
model.summary()

# Visualização Gráfica da Arquitetura (requer pydot e graphviz)
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

# Treinar o modelo e guardar o histórico
history = model.fit(train_images, train_labels_one_hot, epochs=10, batch_size=128, validation_split=0.2)

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
