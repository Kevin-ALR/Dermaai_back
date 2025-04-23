# API de Clasificación de Lesiones Primarias en la Piel

Esta API REST desarrollada en Flask permite clasificar imágenes de lesiones primarias en la piel en cuatro categorías: mancha, roncha, ampolla y pústula, utilizando un modelo de aprendizaje profundo basado en ResNet50.

## Características

- Clasificación de imágenes de lesiones en 4 categorías.
- Retorno de predicciones con porcentajes de probabilidad para cada clase.
- Interfaz REST sencilla para integración con diferentes plataformas.
- Compatible con imágenes en varios formatos (JPG, PNG, etc).

## Requisitos previos

- Python 3.7 o superior
- pip (gestor de paquetes de Python)
- Modelo entrenado

## Instalación

Clonar el repositorio o descargar los archivos:

```bash
git clone <url-del-repositorio>
cd <nombre-del-repositorio>
```

## Instalar las dependencias:

```bash 
pip install flask torch torchvision pillow
```

## Endpoints
1. Verificar estado de la API
GET /health
Respuesta:
json{
  "status": "API en funcionamiento",
  "model_loaded": true
}
2. Clasificar una imagen de lesión
POST /classify
Parámetros:

image-to-clasify: Archivo de imagen (form-data)

## Ejemplo de uso con curl:
bashcurl -X POST -F "image-to-clasify=@ruta/a/tu/imagen.jpg" http://localhost:5000/classify
Ejemplo de uso con Python Requests:
pythonimport requests

url = "http://localhost:5000/classify"
files = {"image-to-clasify": open("ruta/a/tu/imagen.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
## Ejemplo de respuesta:
```
json{
  "predicted_class": "mancha",
  "probabilities": {
    "mancha": 85.72,
    "roncha": 8.45,
    "ampolla": 4.12,
    "pustula": 1.71
  }
}
```