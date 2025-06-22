import os
import matplotlib.pyplot as plt

import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
from PIL import Image
import numpy as np


# Carregar modelo YOLOv8
model = YOLO("yolov8n.pt")  # Modelo leve

def visualizar_imagem(img, results, show_results=False):
    img = cv2.imread(img) # abre a imagem em formato BGR (azul, verde, vermelho). gera a saida (478, 848, 3). que representa a altura, largura e os canais da imagem 
    #print(np.shape(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # converte a imagem de formato BGR para RGB para ser utilizado no matplotlib
      
    # Plotamos a imagem com as detecções
    plt.figure(figsize=(10, 10)) # Cria uma nova figura (imagem) com tamanho 10x10 polegadas
    plt.imshow(cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)) # gera uma imagem com resultados plotados e converte do BGR para RGB
    plt.axis('off') # remove os eixos
    #plt.show() # mostra a imagem
    
    boxes = results[0].boxes
    
    if len(boxes) > 0:
        print(f"Foram detectados {len(boxes)} objetos")
    else:
        print("Nenhum pombo detectado")
        
    if show_results:
        print("\nInformações específicas do resultado:")
            # Exibimos cada detecção
        for i, box in enumerate(boxes):
            cls = int(box.cls[0])
            name = model.names[cls]
            conf = float(box.conf[0])
            print(f"Detecção {i+1}: {cls} - {name} com confiança {conf:.2f}")

        
    
# realiza a inferência

img_testes=['assets/outros.png','assets/pombo.jpg','assets/pombo2.jpg']

# realizando inferência
results = model(
   
    img_testes[2],  # imagem de verificação
    classes=[14], # filtra a classe que será detectada.
    )



# Visualizar os resultados
visualizar_imagem("assets/pombo.jpg", results, True)