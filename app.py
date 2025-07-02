import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from detectores.detectores import detect_image, detector_celular, detect_camera

# Título do app
st.title("Detector de Pombos")

# Opção de escolha entre imagem ou vídeo
opcao = st.radio("Escolha o tipo de mídia para detectar pombos:", ("Imagem", "Vídeo","Camera"))

# Carregar modelo YOLOv8
model = YOLO("yolov8n.pt")  # Modelo leve

# ====== PROCESSAR IMAGEM ======
if opcao == "Imagem":
    uploaded_image = st.file_uploader("Faça upload de uma imagem", type=["jpg", "jpeg", "png"])
    
    if uploaded_image:
        detect_image(uploaded_image)
        
elif opcao =="Vídeo":
    uploaded_video = st.file_uploader("Faça upload de um vídeo", type=["mp4", "avi", "mov"])
    
    if uploaded_video:
        # detect_video(uploaded_video)  # Chama a função de detecção
        # Caminho para o vídeo AVI
        video_path = "./runs/detect/predict/predict.mp4"

        # Exibe o vídeo no app Streamlit
        st.video(video_path)

else:
    detect_camera()
    #detector_celular()
       
#