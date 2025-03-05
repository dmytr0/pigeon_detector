#!/usr/bin/env python3
import os
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

# Визначаємо базовий каталог (де знаходиться цей скрипт)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Каталог з моделями та лейблами
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Шляхи до файлів моделей і лейблів у каталозі models
DETECTION_MODEL = os.path.join(MODELS_DIR, 'ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite')
DETECTION_LABELS = os.path.join(MODELS_DIR, 'coco_labels.txt')
CLASSIFIER_MODEL = os.path.join(MODELS_DIR, 'mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite')
CLASSIFIER_LABELS = os.path.join(MODELS_DIR, 'inat_bird_labels.txt')

def load_labels(path):
    labels = {}
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                parts = line.split(maxsplit=1)
                if len(parts) == 2 and parts[0].isdigit():
                    labels[int(parts[0])] = parts[1]
                else:
                    labels[i] = line
    return labels

# Завантаження лейблів
det_labels = load_labels(DETECTION_LABELS)
cls_labels = load_labels(CLASSIFIER_LABELS)

# Цільові лейбли для голубів (порівняння ведеться в нижньому регістрі)
target_pigeon_labels = ['columba livia domestica', 'columba livia']

def init_interpreter(model_path):
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def main():
    # Ініціалізація інтерпретаторів для детекції та класифікації
    det_interpreter = init_interpreter(DETECTION_MODEL)
    cls_interpreter = init_interpreter(CLASSIFIER_MODEL)

    # Деталі для моделі детекції
    det_input_details = det_interpreter.get_input_details()
    det_output_details = det_interpreter.get_output_details()
    det_height = det_input_details[0]['shape'][1]
    det_width = det_input_details[0]['shape'][2]

    # Деталі для моделі класифікації
    cls_input_details = cls_interpreter.get_input_details()
    cls_output_details = cls_interpreter.get_output_details()
    cls_height = cls_input_details[0]['shape'][1]
    cls_width = cls_input_details[0]['shape'][2]

    # Відкриття камери
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Не вдалося відкрити камеру")
        return

    print("Запуск розпізнавання. Натисніть 'q' для виходу.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Не вдалося зчитати кадр")
            break

        # Підготовка зображення для детекції
        det_frame = cv2.resize(frame, (det_width, det_height))
        if det_input_details[0]['dtype'] == np.uint8:
            input_det = np.expand_dims(det_frame, axis=0).astype(np.uint8)
        else:
            input_det = np.expand_dims(det_frame, axis=0).astype(np.float32) / 255.0

        # Запуск детекції
        det_interpreter.set_tensor(det_input_details[0]['index'], input_det)
        det_interpreter.invoke()

        # Отримання результатів детекції
        boxes = det_interpreter.get_tensor(det_output_details[0]['index'])[0]      # [N, 4]
        classes = det_interpreter.get_tensor(det_output_details[1]['index'])[0]    # [N]
        scores = det_interpreter.get_tensor(det_output_details[2]['index'])[0]     # [N]

        imH, imW, _ = frame.shape

        for i in range(len(scores)):
            if scores[i] < 0.5:
                continue
            class_id = int(classes[i])
            label = det_labels.get(class_id, '').lower()
            # Розглядаємо лише об'єкти, що містять слово "bird"
            if 'bird' not in label:
                continue

            # Перетворення координат
            ymin, xmin, ymax, xmax = boxes[i]
            x1 = int(xmin * imW)
            y1 = int(ymin * imH)
            x2 = int(xmax * imW)
            y2 = int(ymax * imH)

            # Вирізання області для класифікації
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            roi_resized = cv2.resize(roi, (cls_width, cls_height))
            if cls_input_details[0]['dtype'] == np.uint8:
                input_cls = np.expand_dims(roi_resized, axis=0).astype(np.uint8)
            else:
                input_cls = np.expand_dims(roi_resized, axis=0).astype(np.float32) / 255.0

            cls_interpreter.set_tensor(cls_input_details[0]['index'], input_cls)
            cls_interpreter.invoke()
            cls_output = cls_interpreter.get_tensor(cls_output_details[0]['index'])[0]
            predicted_index = np.argmax(cls_output)
            predicted_confidence = cls_output[predicted_index]
            predicted_label = cls_labels.get(predicted_index, '').lower()

            if predicted_confidence > 0.5 and any(pigeon in predicted_label for pigeon in target_pigeon_labels):
                color = (0, 255, 0)  # зелений для голуба
                text = f"Pigeon {predicted_confidence:.2f}"
            else:
                color = (0, 0, 255)  # червоний для інших
                text = f"Not Pigeon {predicted_label[:15]} {predicted_confidence:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Pigeon Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()