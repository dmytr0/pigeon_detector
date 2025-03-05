import cv2
import numpy as np
import time
from tflite_runtime.interpreter import Interpreter

# Шляхи до моделі та файлу з мітками (labels)
MODEL_PATH = 'pigeon.tflite'
LABELS_PATH = 'labels.txt'


# Функція для завантаження міток із файлу
def load_labels(path):
    labels = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                # Припускаємо, що формат: id label
                pair = line.strip().split(maxsplit=1)
                if len(pair) == 2:
                    labels[int(pair[0])] = pair[1]
    return labels


labels = load_labels(LABELS_PATH)

# Ініціалізація TFLite-інтерпретатора для роботи на CPU
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Отримання деталей моделі (інформація про вхід та вихід)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Припускаємо, що модель приймає зображення розміром, наприклад, (300, 300)
input_shape = input_details[0]['shape']
height = input_shape[1]
width = input_shape[2]

# Поріг впевненості для відображення детекції
CONF_THRESHOLD = 0.5

# Ініціалізація камери (зверніть увагу, що для Pi Camera можна використати бібліотеку picamera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Не вдалося отримати кадр з камери")
        break

    # Збереження копії кадру для відображення
    output_frame = frame.copy()

    # Зміна розміру кадру до розміру, який приймає модель
    resized_frame = cv2.resize(frame, (width, height))

    # Передобробка: розширення розмірності та нормалізація (якщо модель тренувалася з нормалізованими пікселями)
    input_data = np.expand_dims(resized_frame, axis=0).astype(np.float32)
    input_data = input_data / 255.0  # нормалізація до [0, 1]

    # Запуск інференсу
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Отримання результатів:
    # Зазвичай об'єктні детектори повертають:
    #   boxes: координати прямокутників (ymin, xmin, ymax, xmax) – нормалізовані значення
    #   classes: id класів
    #   scores: значення впевненості
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # [num_detections, 4]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # [num_detections]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # [num_detections]

    imH, imW, _ = frame.shape

    # Обробка кожної детекції
    for i in range(len(scores)):
        if scores[i] >= CONF_THRESHOLD:
            # Якщо у вашій моделі для голуба використовується певний id (наприклад, 0 або 1), можна перевірити:
            class_id = int(classes[i])
            label = labels.get(class_id, 'Pigeon')
            # Якщо модель тренувалась лише на голубах, ця перевірка може бути не обов’язковою.
            # Отримання координат прямокутника
            ymin, xmin, ymax, xmax = boxes[i]
            x1 = int(xmin * imW)
            y1 = int(ymin * imH)
            x2 = int(xmax * imW)
            y2 = int(ymax * imH)

            # Малювання прямокутника та напису з міткою і впевненістю
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(output_frame, f'{label} {scores[i]:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Відображення кадру з результатами
    cv2.imshow('Розпізнавання голубів', output_frame)

    # Вихід з циклу при натисканні 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
