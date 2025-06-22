import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from detectores.detectores import detect_image, detect_video
from glob import glob

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
    
    #if uploaded_video:
        # Salva o vídeo em um arquivo temporário
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    #tfile.write(uploaded_video.read())
    #tfile.close()  # Garante que o arquivo seja salvo antes de usar

    # Executa a detecção com o YOLO
    # model(tfile.name, conf=0.3, classes=[14], save=True)

    # Mostra mensagem no app
    st.success("Detecção concluída. Vídeo processado com sucesso!")

    # Exibe vídeo anotado se YOLO salvou
    output_path = os.path.join("runs", "detect", os.listdir("runs/detect")[-1], os.path.basename(tfile.name))
    output_videos = glob('runs/detect/redict/tmpxgcmca4j.mp4')
    st.success(output_path)
    st.info(output_videos)
    #st.video(output_videos[0])
        # if os.path.exists(output_path):
    st.video('runs/detect/redict/tmpxgcmca4j.mp4')
        # else:
        #     st.warning("Não foi possível exibir o vídeo de saída.")
        #detect_video(uploaded_video)  # Chama a função de detecção
       
# ====== PROCESSAR VÍDEO ======
# else:
#     uploaded_video = st.file_uploader("Faça upload de um vídeo", type=["mp4", "avi", "mov"])

#     if uploaded_video is not None:
#         # Salvar vídeo temporariamente
#         tfile = tempfile.NamedTemporaryFile(delete=False)
#         tfile.write(uploaded_video.read())
#         video_path = tfile.name

#         # Abrir vídeo
#         cap = cv2.VideoCapture(video_path)
#         fps = int(cap.get(cv2.CAP_PROP_FPS))
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#         # Criar arquivo de saída
#         fourcc = cv2.VideoWriter_fourcc(*"XVID")
#         output_path = os.path.join(tempfile.gettempdir(), "output.avi")
#         out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#         # Status de detecção
#         pombo_detectado_em_algum_frame = False
#         st.text("Processando vídeo...")

#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             results = model(frame)
#             pombo_detectado_no_frame = False

#             for result in results:
#                 for box in result.boxes:
#                     cls = int(box.cls[0])
#                     if cls == 14:  # Classe "bird"
#                         x1, y1, x2, y2 = map(int, box.xyxy[0])
#                         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                         cv2.putText(frame, "Pombo", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
#                         pombo_detectado_no_frame = True

#             if pombo_detectado_no_frame:
#                 pombo_detectado_em_algum_frame = True

#             out.write(frame)

#         cap.release()
#         out.release()

#         # Exibir vídeo com detecções
#         st.success("Processamento concluído!")
#         st.video(output_path)

#         # Mensagem abaixo do vídeo
#         if pombo_detectado_em_algum_frame:
#             st.markdown("### ✅ Pombo encontrado")
#         else:
#             st.markdown("### ❌ Pombo não encontrado")
