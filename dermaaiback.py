from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import io

app = Flask(__name__)

# Definición de las clases de lesiones
CLASSES = ['mancha', 'roncha', 'ampolla', 'pustula']
OUTPUT_DIM = len(CLASSES)

# Configuración para las transformaciones de la imagen
pretrained_size = 224
pretrained_means = [0.485, 0.456, 0.406]
pretrained_stds = [0.229, 0.224, 0.225]

# Transformaciones para las imágenes de entrada
image_transforms = transforms.Compose([
    transforms.Resize(pretrained_size),
    transforms.CenterCrop(pretrained_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=pretrained_means, std=pretrained_stds)
])

# Definición del modelo ResNet con funcionalidad para extraer características
class ResNetWithFeatures(nn.Module):
    def __init__(self, resnet_model, dropout_p=0.8):
        super().__init__()
        self.resnet = resnet_model
        
        # Extraer características hasta layer4
        self.features = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4
        )
        self.avgpool = self.resnet.avgpool
        
        # Capa FC con Dropout
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(self.resnet.fc.in_features, OUTPUT_DIM)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        h = torch.flatten(x, 1)  # Características
        x = self.fc(h)           # Predicciones (con Dropout aplicado)
        return x, h

# Inicializa el modelo
def load_model():
    # Determinar el dispositivo disponible
    device = torch.device('cpu') 
    print(f"Usando dispositivo: {device}")
    
    base_model = models.resnet50(pretrained=True)
    
    num_ftrs = base_model.fc.in_features
    base_model.fc = nn.Linear(num_ftrs, OUTPUT_DIM)
    
    model = ResNetWithFeatures(base_model, dropout_p=0.8)
    
    try:
        model.load_state_dict(torch.load('models/modelo_94%.pt', map_location=device))
        print("Modelo cargado correctamente")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
    
    model = model.to(device)
    model.eval()
    return model, device

# Cargar el modelo al iniciar la aplicación
model, device = load_model()

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'image-to-clasify' not in request.files:
        return jsonify({'error': 'No se recibió ninguna imagen'}), 400
    
    file = request.files['image-to-clasify']
    
    if file.filename == '':
        return jsonify({'error': 'Nombre de archivo vacío'}), 400
    
    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        img_tensor = image_transforms(img).unsqueeze(0).to(device)
        
        # Realizar la predicción
        with torch.no_grad():
            predictions, _ = model(img_tensor)
            probabilities = torch.nn.functional.softmax(predictions, dim=1)[0]
        
        probs = probabilities.cpu().tolist()  
        
        # Crear el resultado
        result = {
            'predicted_class': CLASSES[probabilities.argmax().item()],
            'probabilities': {CLASSES[i]: round(probs[i] * 100, 2) for i in range(len(CLASSES))}
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Ruta de prueba para verificar que la API está funcionando
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'API en funcionamiento', 'model_loaded': model is not None})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)