import os
import cv2
import numpy as np
import base64
import re
from flask import Flask
from flask_socketio import SocketIO
import threading
from ultralytics import YOLO

# Carrega o modelo a partir do caminho local dentro do contêiner
model_path = './models/yolov8n.pt' # Ou '/app/models/yolov8n.pt'
model = YOLO(model_path)

# ====== SUA FUNÇÃO DE VISÃO COMPUTACIONAL ======

blur = 3 
retangulo = []
cm_px = 30

def processamento_imagem(frame, id):
    global blur, cm_px, retangulo

    # Convertendo para escala de cinza
    imagem_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Equalização de histograma para melhorar o contraste
    imagem_equalizada = cv2.equalizeHist(imagem_cinza)

    # Aplicar desfoque para reduzir ruído
    imagem_blur = cv2.GaussianBlur(imagem_equalizada, (blur, blur), 0)

    # Detectar bordas com Canny (opcional, pode ser substituído por limiarização)
    bordas = cv2.Canny(imagem_blur, 100, 150)

    # Encontrar contornos
    contornos, _ = cv2.findContours(bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    medidas = []

    for contorno in contornos:
        # Ignorar pequenos contornos
        if cv2.contourArea(contorno) > 500:
            # Desenhar linhas conectando os pontos do contorno
            for i in range(len(contorno)):
                ponto1 = tuple(contorno[i][0])
                ponto2 = tuple(contorno[(i + 1) % len(contorno)][0])
                cv2.line(frame, ponto1, ponto2, (0, 255, 0), 2)
                

            # Retângulo envolvente
            x, y, w, h = cv2.boundingRect(contorno)
            cm_px = 1 if cm_px == 0 else cm_px
            cm_px = 39
            cm_w = w / cm_px
            cm_h = h / cm_px
            cv2.putText(frame, f"{(cm_w):.2f} x {(cm_h):.2f} ", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            medidas.append({"largura_cm": round(cm_w, 2), "altura_cm": round(cm_h, 2)})

    # Codifica em base64
    _, buffer = cv2.imencode('.jpg', frame)
    frame_b64 = base64.b64encode(buffer).decode('utf-8')
    socketio.emit("server_frame", {"cameraId": id,"frame": f"data:image/jpeg;base64,{frame_b64}", "medidas": medidas})
    
def processamento_imagem_yolo(frame, id):
    # Detecta objetos no frame
    results = model(frame)
    
    medidas = []

    for result in results:
        for box in result.boxes:
            # Coordenadas da caixa delimitadora
            x1, y1, x2, y2 = box.xyxy[0]  # canto sup. esquerdo e inf. direito
            largura_px = float(x2 - x1)
            altura_px = float(y2 - y1)

            # Converte para cm (se calibrado)
            largura_cm = largura_px / cm_px
            altura_cm = altura_px / cm_px

            # Classe e confiança
            classe = int(box.cls[0])

            # Nome da classe detectada
            nome_classe = model.names[classe]

            # Exibe informações no frame
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame,
                        f"{largura_cm:.2f}cm x {altura_cm:.2f}cm",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            medidas.append([round(largura_cm, 2),  round(altura_cm, 2)])

    # Codifica em base64
    _, buffer = cv2.imencode('.jpg', frame)
    frame_b64 = base64.b64encode(buffer).decode('utf-8')
    socketio.emit("server_frame_yolo", {"cameraId": id,"frame": f"data:image/jpeg;base64,{frame_b64}", "medidas": medidas})


# ====== CONFIGURAÇÃO FLASK ======
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')


# ====== RECEBENDO FRAMES EM TEMPO REAL ======
@socketio.on("frame")
def handle_frame(data):
    try:
        # Remove prefixo "data:image/jpeg;base64,"
        img_data = re.sub('^data:image/.+;base64,', '', data["data"])
        img_bytes = base64.b64decode(img_data)

        # Converte bytes para numpy array (imagem OpenCV)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            print("Frame inválido recebido")
            return

        camera_id = data.get("cameraId")
        
        # Processa e envia frame
        processamento_imagem_yolo(frame, camera_id)

    except Exception as e:
        print("Erro ao processar frame:", e)
        
# ===== NOVO EVENTO: RECEBER IMAGENS PARA TREINO =====
@socketio.on("upload_image")
def handle_upload_image(data, nome):
    """
    Recebe imagem base64 + classe e salva no dataset (train ou val)
    """
    try:
        image_b64 = data.get("image")
        class_name = data.get("class", nome)   # nome da pasta da classe
        dataset_type = data.get("type", "train")    # train ou val
        filename = data.get("filename", f"{nome}.jpg")  # nome opcional

        # Define caminho
        save_dir = os.path.join("datasets", dataset_type, class_name)
        os.makedirs(save_dir, exist_ok=True)

        # Remove prefixo base64 e salva
        img_data = re.sub("^data:image/.+;base64,", "", image_b64)
        img_bytes = base64.b64decode(img_data)
        img_path = os.path.join(save_dir, filename)

        with open(img_path, "wb") as f:
            f.write(img_bytes)

        socketio.emit("upload_status", {"status": "ok", "file": img_path})

    except Exception as e:
        socketio.emit("upload_status", {"status": "error", "error": str(e)})


# ====== FECHAR JANELA QUANDO CLIENTE DESCONECTAR ======
@socketio.on("disconnect")
def handle_disconnect():
    print("Cliente desconectado")

if __name__ == "__main__":
    # O uso do eventlet no requirements.txt torna o `allow_unsafe_werkzeug` desnecessário
    socketio.run(app, host="0.0.0.0", port=5000)
