from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import os                           # Manipulação de arquivos do sistema de arquivos
import pandas as pd                 # Permite ler o arquivo results.csv gerado pelo treinamento e extrair métricas a partir dos dados
import seaborn as sns               # Biblioteca para visualização de dados baseada no Matplotlib, com melhorias de visuais

# =========================================
# 1. Criar pasta de saída
# =========================================
output_dir = '/home/user/Documentos/PPgTI/ComputerVision/Projeto/plot'
os.makedirs(output_dir, exist_ok=True)

# =========================================
# 2. Configurar caminhos do dataset e modelo
# =========================================
dataset_path = '/home/user/Documentos/PPgTI/ComputerVision/Projeto/datasets/merged_crack_det.v1i.yolov11'
data_yaml_path = f'{dataset_path}/data.yaml'

# =========================================
# 3. Carregar e treinar o modelo
# =========================================
model = YOLO("yolo11n.pt")  # ou yolo11n.pt se quiser mais leve

results = model.train(
    data=data_yaml_path,
    epochs=200,
    imgsz=640,
    batch=8,
    project="detect_cracks",
    name="yolo11n_cracks"
)

# =========================================
# 4. Plotar curva de acurácia e precisão
# =========================================
log_path = "detect_cracks/yolo11n_cracks/results.csv"

if os.path.exists(log_path):
    df = pd.read_csv(log_path)

    plt.figure(figsize=(10, 5))
    plt.plot(df["epoch"], df["metrics/precision(B)"], label="Precision")
    plt.plot(df["epoch"], df["metrics/recall(B)"], label="Recall")
    plt.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP@0.5")
    plt.xlabel("Época")
    plt.ylabel("Métrica")
    plt.title("Curva de Acurácia e Precisão")
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_dir}/curva_acuracia.png")
    plt.close()
else:
    print(f"[!] CSV de resultados não encontrado em {log_path}")

# =========================================
# 5. Avaliação final + matriz de confusão
# =========================================
val_results = model.val()
conf_matrix = val_results.confusion_matrix.matrix
labels = val_results.names

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt=".0f", cmap="Blues",
            xticklabels=labels.values(), yticklabels=labels.values())
plt.xlabel("Predito")
plt.ylabel("Real")
plt.title("Matriz de Confusão")
plt.savefig(f"{output_dir}/matriz_confusao.png")
plt.close()

# =========================================
# 6. Detecção em imagem de teste
# =========================================
test_image_path = f'{dataset_path}/test/images/crack-366-_jpg.rf.4e18ec595afcdfda0a4798b9961e154f.jpg'
results = model(test_image_path)

img = cv2.imread(test_image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
result_img = results[0].plot()
result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

fig, axs = plt.subplots(1, 2, figsize=(15, 7))
axs[0].imshow(img)
axs[0].set_title('Imagem Original')
axs[0].axis('off')

axs[1].imshow(result_img_rgb)
axs[1].set_title('Detecção de Rachaduras')
axs[1].axis('off')

plt.tight_layout()
plt.savefig(f"{output_dir}/detecao_rachadura.png")
plt.show()
