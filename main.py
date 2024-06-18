import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from PIL import Image

# Função para redimensionar as imagens
def resize_images(input_folder, output_folder, size=(224, 224)):
    # Verifique se a pasta de saída existe, se não, crie-a
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Iterar por todos os arquivos na pasta de entrada
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)
            img_resized = img.resize(size, Image.LANCZOS)  # Usando Image.LANCZOS para redimensionamento
            output_path = os.path.join(output_folder, filename)
            img_resized.save(output_path)

# Caminho para a pasta com as imagens originais
input_folder = 'imagens'

# Caminho para a pasta onde as imagens redimensionadas serão salvas
output_folder = 'imagensalteradas'

# Redimensionar as imagens
resize_images(input_folder, output_folder)

# Carregar o modelo VGG16 pré-treinado
model = VGG16(weights='imagenet')

# Função para carregar e pré-processar uma imagem
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Função para fazer a predição e decodificar os resultados
def predict_image(model, img_path):
    img_array = load_and_preprocess_image(img_path)
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    return decoded_predictions

# Listar todas as imagens redimensionadas na pasta
img_files = [f for f in os.listdir(output_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Processar e exibir cada imagem com suas predições
for img_file in img_files:
    img_path = os.path.join(output_folder, img_file)
    predictions = predict_image(model, img_path)

    # Carregar e exibir a imagem
    img = image.load_img(img_path, target_size=(224, 224))
    plt.imshow(img)
    plt.axis('off')
    plt.title(img_file)
    plt.show()

    # Exibir as predições
    print(f"Predictions for {img_file}:")
    for pred in predictions:
        print(f"{pred[1]}: {pred[2]*100:.2f}%")
    print("\n")
