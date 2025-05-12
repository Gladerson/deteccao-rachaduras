# deteccao-rachaduras

# ESTRUTURA DE DIRETÓRIOS:
home/user/Documentos/PPgTI/ComputerVision/Projeto/
├── Python_virtual_environment_of_the_PPgTI_computer_vision_project/
├── detectar_rachaduras.py                          ← script principal de inferência
├── /datasets/
│   └── merged_crack_det.v1i.yolov11/               ← https://universe.roboflow.com/crack-detection-dataset/merged_crack_det/dataset/1
│       ├── data.yaml
│       ├── train/
│       ├── valid/
│       ├── test/
│       └── detect_cracks/
│           └── yolo11n_cracks/
│               └── weights/
│                   └── best.pt                     ← modelo treinado
├── /plot/                                          ← saída de gráficos do treinamento e imagens de teste
└── /img_testes/
    └── 01.jpg                                      ← imagem de estrada usada para teste

# 1. Instalar Python
sudo apt update
sudo apt install python3 python3-venv python3-pip -y

# 2. Criar ambiente virtual
cd /home/user/Documentos/PPgTI/ComputerVision/Projeto
python3 -m venv virtual_environment
source virtual_environment/bin/activate

# 3. Instalar pacotes no ambiente
pip install --upgrade pip
pip install numpy pillow pandas matplotlib seaborn
pip install opencv-python
pip install scikit-image

# 4. Instalar YOLO com uma TEMP alternativa, porque a padrão do sistema possui limite de armazenamento
mkdir -p ~/tmp_pip
export TMPDIR=~/tmp_pip
pip install ultralytics
rm -rf ~/tmp_pip
unset TMPDIR

# 2. Limpa cache do ambiente virtual
pip cache purge
