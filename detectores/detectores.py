
import cv2
import tempfile
import streamlit as st
from ultralytics import YOLO
import os
import time
import pygame

import threading


# Inicializa o pygame para tocar som
pygame.mixer.init()
gaviao_som = "gaviao.mp3"  # coloque o nome do arquivo de som aqui

model = YOLO("yolov8n.pt")  # Modelo leve


# Inicializa o pygame
pygame.mixer.init()

# Caminho do som
gaviao_som = "gaviao.mp3"  # Troque isso se estiver em outra pasta

def tocar_som_loop():
    if not pygame.mixer.music.get_busy():
        try:
            if os.path.exists(gaviao_som):
                pygame.mixer.music.load(gaviao_som)
                pygame.mixer.music.play(-1)
            else:
                st.error(f"Arquivo de som não encontrado: {gaviao_som}")
        except Exception as e:
            st.error(f"Erro ao tocar som: {e}")

def parar_som():
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()

def detect_image(path_img, result=True):
    # Salvar a imagem em um arquivo temporário
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(path_img.read())
        
        # Ler a imagem com OpenCV
        img = cv2.imread(tfile.name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # converte a imagem de formato BGR para RGB para ser utilizado no matplotlib

        # realiza a inferência
        preditc = model(img, classes=[14])
        boxes = preditc[0].boxes
        
        annotated_img = preditc[0].plot()  # Imagem com bounding boxes
        st.image(annotated_img, caption="Imagem com detecções", use_container_width=True)
        
        if result:
            if len(boxes) > 0:
                st.markdown(f"### ✅ Foram encontrados {len(boxes)} objetos")
                st.success(f"Emitir som do gavião")
        
                st.success("Informações específicas do resultado:")
                    # Exibimos cada detecção
                for i, box in enumerate(boxes):
                    cls = int(box.cls[0])
                    name = model.names[cls]
                    conf = float(box.conf[0])
                    st.info(f"Detecção {i+1}: classe {cls}( {name}) com confiança {conf:.2f}")
            else:
                st.warning("### ❌ Pombo não encontrado")
            
        return preditc
   

def detect_video(path_video):
    # Salva o vídeo em um arquivo temporário
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(path_video.read())
    tfile.close()  # Garante que o arquivo seja salvo antes de usar
    capture_video = cv2.VideoCapture(tfile.name)
    nome_video = 'resultado.mp4'
    
        # Define o codec MP4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
    fps = int(capture_video.get(cv2.CAP_PROP_FPS))
    width = int(capture_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)
    
    saida_video = cv2.VideoWriter(nome_video, fourcc, fps, frame_size)

     # Aqui é onde você poderia aplicar um modelo frame a frame se necessário
    while True:
        ret, frame = capture_video.read()
        if not ret:
            break

        # Aqui você pode aplicar alguma detecção no frame
        # Por exemplo: frame = aplicar_modelo(frame)

        saida_video.write(frame)

    # Libera os recursos
    capture_video.release()
    saida_video.release()


    # Executa a detecção com o YOLO
    model(tfile.name, conf=0.3, classes=[14], save=True)

    # Mostra mensagem no app
    st.success("Detecção concluída. Vídeo processado com sucesso!")

    # Exibe vídeo anotado se YOLO salvou
    output_path = os.path.join("runs", "detect", os.listdir("runs/detect")[-1], os.path.basename(tfile.name))
    if os.path.exists(output_path):
        st.video(output_path)
    else:
        st.warning("Não foi possível exibir o vídeo de saída.")
        

def detect_camera():
  
    # Cria espaço para mostrar os frames e mensagens
    frame_placeholder = st.empty()
    message_placeholder = st.empty()

    # Inicializar a câmera (use 0 ou 1, dependendo da sua webcam)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Erro ao abrir a câmera")
        return

    # Botão para parar
    stop = st.button("Parar")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Erro ao capturar o frame")
            break

        # Realizar detecção (classe 0 = pessoa)
        results = model(frame, conf=0.3, classes=[14])
        boxes = results[0].boxes

        # Verifica se encontrou alguma pessoa
        if boxes and len(boxes) > 0:
            message_placeholder.success("Pombo encontrada!")
            tocar_som_loop()  # ▶️ toca o som em loop
        else:
            message_placeholder.warning("Pombo não encontrada.")
            parar_som()  # ⏹️ para o som se estiver tocando

        # Desenhar as detecções no frame
        annotated_frame = results[0].plot()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Exibir o frame no Streamlit
        frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)

        if stop:
            break

        time.sleep(0.03)

    cap.release()
    st.success("Câmera encerrada.")

def detector_celular():
    ip = 'https://192.168.15.4:8080/video'  # Substitua pelo IP da sua câmera
            
    # Cria espaço para mostrar os frames e mensagens
    frame_placeholder = st.empty()
    message_placeholder = st.empty()

    #Inicializar a câmera (use 0 ou 1, dependendo da sua webcam)
    cap = cv2.VideoCapture()
    cap.open(ip)

    if not cap.isOpened():
        st.error("Erro ao abrir a câmera")
        return

    # Botão para parar
    stop = st.button("Parar")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Erro ao capturar o frame")
            break

        # Realizar detecção (classe 0 = pessoa)
        results = model(frame, conf=0.3, classes=[])
        boxes = results[0].boxes

        # Verifica se encontrou alguma pessoa
        if boxes and len(boxes) > 0:
            message_placeholder.success("Pessoa encontrada!")
        else:
            message_placeholder.warning("Pessoa não encontrada.")

        # Desenhar as detecções no frame
        annotated_frame = results[0].plot()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Exibir o frame no Streamlit
        frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)

        if stop:
            break

        time.sleep(0.03)

    cap.release()
    st.success("Câmera encerrada.")