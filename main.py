import cv2
import numpy as np

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
lbph = cv2.face.LBPHFaceRecognizer_create()

# pede o nome da pessoa
name = input("Digite o nome da pessoa: ")

# variável para verificar se o treinamento já foi realizado
trained = False

while True:

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detecta as faces na imagem
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        # desenha um retângulo na face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        # extrai a região da face
        roi_gray = gray[y:y+h, x:x+w]

        # realiza o treinamento se ainda não foi realizado
        if not trained:
            # adiciona a face ao conjunto de treinamento
            face = cv2.resize(roi_gray, (200, 200))
            lbph.train([face], np.array([1]))
            trained = True

        # realiza a predição da face
        face_pred, confidence = lbph.predict(cv2.resize(roi_gray, (200, 200)))

        # se a face for reconhecida, exibe o nome, senão exibe 'Desconhecido'
        if confidence < 100:
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, 'Pessoa Desconhecida', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

    # exibe o frame na janela
    cv2.imshow('frame',frame)

    # encessa o loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()