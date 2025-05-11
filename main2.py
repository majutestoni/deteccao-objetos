import cv2  # Biblioteca OpenCV para processar imagens e vídeos
from ultralytics import YOLO  # Importa a classe do modelo YOLO da biblioteca Ultralytics
from google.colab.patches import cv2_imshow
import time  # Biblioteca para pausar a execução (usada para dar tempo de ver os frames)

# Carrega o modelo YOLOv8 versão "nano" (leve e rápido), já pré-treinado
# model = YOLO('yolov8m.pt')
model = YOLO('yolov8x.pt')  # maior e mais preciso



# Definição das classes que iremos focar
target_classes = {'car', 'truck'}
car_count = 0
truck_count = 0

# Linha para realizarmos a contagem
line_y = 300  # Posição vertical da linha
offset = 40   # Margem de tolerância para a linha

# Abre o arquivo de vídeo para leitura
cap = cv2.VideoCapture("fast motion cars moving on highway.mp4")

counted_ids = set()
frame_count = 0  # Contador de frames (quadros do vídeo)

class_names = model.model.names

# Enquanto o vídeo estiver aberto (não acabou nem deu erro)
while cap.isOpened():
    ret, frame = cap.read()  # Lê o próximo frame do vídeo

    if not ret:
        break  # Sai do loop se não conseguiu ler (fim do vídeo, por exemplo)

    frame_count += 1  # Incrementa o contador de frames

    # Só processa e mostra a cada 5 frames
    if frame_count % 10 == 0:
        # Aplica a detecção de objetos no frame atual
        results = model.predict(source=frame, imgsz=736, conf=0.55, verbose=False)
        # results = model.predict(source=frame, imgsz=736, conf=0.5, verbose=False)
        detections = results[0].boxes
        annotated_frame = frame.copy()

        cv2.line(annotated_frame, (0, line_y), (annotated_frame.shape[1], line_y), (0, 255, 255), 2)
        if detections is not None:
            for i, box in enumerate(detections.xyxy.cpu().numpy()):
                x1, y1, x2, y2 = map(int, box[:4])
                class_id = int(detections.cls[i].cpu().numpy())
                label = class_names[class_id]
                print(label)  # <--- Veja se 'truck' aparece

                if label not in target_classes:
                    continue  # Ignora outras classes

                # Centro do objeto
                # cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cx = (x1 + x2) // 2
                # Desenha o retângulo e centro
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # cv2.circle(annotated_frame, (cx, cy), 3, (0, 0, 255), -1)
                # cx = (x1 + x2) // 2

                if label == 'car':
                  cy = (y1 + y2) // 2  # centro vertical da caixa
                elif label == 'truck':
                  cy = y2  # base da caixa (parte de baixo do caminhão)

                cv2.circle(annotated_frame, (cx, cy), 3, (0, 0, 255), -1)


                # Identificador para evitar dupla contagem
                track_id = f"{cx}-{cy}-{class_id}"
                if (line_y - offset) < cy < (line_y + offset) and track_id not in counted_ids:
                    counted_ids.add(track_id)
                    if label == 'car':
                        car_count += 1
                    elif label == 'truck':
                        truck_count += 1


        cv2.putText(annotated_frame, f"Carros: {car_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Caminhoes: {truck_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

        cv2_imshow(annotated_frame)

        # Dá uma pausa de 0.2 segundos para que a imagem possa ser vista
        time.sleep(0.5)

# Libera o vídeo e fecha as janelas (mesmo que não sejam usadas no Colab)
cap.release()
