from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import os

# =====================================
# 1. Carregar modelo treinado localmente
# =====================================
model_path = "/home/user/Documentos/PPgTI/ComputerVision/Projeto/datasets/merged_crack_det.v1i.yolov11/detect_cracks/yolo11n_cracks/weights/last.pt"
model = YOLO(model_path)

# =====================================
# 2. Caminho da imagem de teste
# =====================================
img_path = "/home/user/Documentos/PPgTI/ComputerVision/Projeto/img_testes/02.jpg"
output_dir = "/home/user/Documentos/PPgTI/ComputerVision/Projeto/plot"

# Verificar se imagem existe
if not os.path.exists(img_path):
    raise FileNotFoundError(f"Imagem não encontrada em {img_path}")

# Criar diretório de saída se não existir
os.makedirs(output_dir, exist_ok=True)

# =====================================
# 3. Leitura e detecção
# =====================================
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

results = model(img_path)

# Plotar resultado da detecção
result_img = results[0].plot()
result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

# =====================================
# 4. Salvar imagem resultante
# =====================================
output_path = os.path.join(output_dir, "resultado_01.jpg")
cv2.imwrite(output_path, cv2.cvtColor(result_img_rgb, cv2.COLOR_RGB2BGR))  # salvar com formato correto de cor

# =====================================
# 5. Exibir imagem com matplotlib
# =====================================
plt.figure(figsize=(10, 8))
plt.imshow(result_img_rgb)
plt.title("Detecção de Rachaduras")
plt.axis('off')
plt.show()
