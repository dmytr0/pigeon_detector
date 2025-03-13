#!/usr/bin/env python3
import os
import cv2
import numpy as np
import tensorflow as tf
import argparse
import random
from tflite_runtime.interpreter import Interpreter

# Вимикаємо eager execution для сумісності із TF1 (детекційна модель)
tf.compat.v1.disable_eager_execution()


def load_detection_model(model_path):
    """Завантажує frozen graph детекційної моделі."""
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


def load_labels(label_path):
    """Завантажує мітки з файлу, де кожен рядок містить назву класу."""
    labels = {}
    with open(label_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):
            labels[i] = line.strip()
    return labels


def run_detection_inference(image, detection_graph):
    """Запускає інференс детекції на зображенні і повертає результати."""
    with detection_graph.as_default():
        with tf.compat.v1.Session() as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            image_expanded = np.expand_dims(image, axis=0)
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_expanded})
            return boxes, scores, classes, int(num_detections[0])


def run_classification(interpreter, input_details, output_details, image):
    """Запускає інференс класифікації на обрізаному зображенні.
       Очікується, що image має потрібний розмір (наприклад, 224x224) і тип uint8."""
    input_data = np.expand_dims(image, axis=0).astype(np.uint8)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data


def draw_detection_and_classification(image, boxes, scores, classes, det_labels, min_score_thresh,
                                      cls_interpreter, cls_input_details, cls_output_details, cls_labels,
                                      num_detections):
    """Для кожного знайденого об'єкта:
       - Малює bounding box.
       - Якщо детекція відповідає класу 'bird' (за COCO), обрізає ROI,
         проводить класифікацію через inat_bird модель, і використовує отриманий
         результат як напис.
       - Для інших об'єктів показує початкову мітку детекції.
       Для тексту використовується контрастне обведення."""
    imH, imW, _ = image.shape
    class_colors = {}
    for i in range(num_detections):
        if scores[0][i] < min_score_thresh:
            continue
        class_id = int(classes[0][i])
        # Отримуємо детекційну мітку з COCO (перетворюємо в нижній регістр для порівняння)
        det_label = det_labels.get(class_id, 'N/A').lower()
        # Обчислюємо координати bounding box
        box = boxes[0][i]
        (ymin, xmin, ymax, xmax) = box
        left = int(xmin * imW)
        top = int(ymin * imH)
        right = int(xmax * imW)
        bottom = int(ymax * imH)

        if class_id not in class_colors:
            class_colors[class_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        color = class_colors[class_id]

        # Якщо детекція відповідає 'bird', запускаємо класифікацію
        if 'bird' in det_label:
            roi = image[top:bottom, left:right]
            if roi.size == 0:
                continue
            # Змінюємо розмір ROI до вхідного розміру класифікаційної моделі
            cls_height = cls_input_details[0]['shape'][1]
            cls_width = cls_input_details[0]['shape'][2]
            roi_resized = cv2.resize(roi, (cls_width, cls_height))
            cls_output = run_classification(cls_interpreter, cls_input_details, cls_output_details, roi_resized)
            predicted_index = np.argmax(cls_output)
            predicted_confidence = cls_output[0][predicted_index]
            predicted_label = cls_labels.get(predicted_index, "Unknown")
            label_text = f"{predicted_label}: {predicted_confidence:.2f}"
        else:
            label_text = f"{det_label}: {scores[0][i]:.2f}"

        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
        # Малюємо текст з обведенням для кращої видимості
        cv2.putText(image, label_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, label_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return image


def main():
    parser = argparse.ArgumentParser(description="Детекція та класифікація птахів")
    parser.add_argument("--image", help="Шлях до зображення", default=None)
    parser.add_argument("--camera", action="store_true", help="Використовувати камеру")
    args = parser.parse_args()

    # Встановлюємо базову директорію та шляхи до файлів (детекція та класифікація)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DET_MODEL_PATH = os.path.join(BASE_DIR, "models", "frozen_inference_graph.pb")
    DET_LABELS_PATH = os.path.join(BASE_DIR, "models", "coco_labels.txt")
    CLS_MODEL_PATH = os.path.join(BASE_DIR, "models", "mobilenet_v2_1.0_224_inat_bird_quant.tflite")
    CLS_LABELS_PATH = os.path.join(BASE_DIR, "models", "inat_bird_labels.txt")

    # Перевірка наявності файлів
    if not os.path.exists(DET_MODEL_PATH):
        print("Не знайдено детекційну модель:", DET_MODEL_PATH)
        return
    if not os.path.exists(DET_LABELS_PATH):
        print("Не знайдено файл детекційних міток:", DET_LABELS_PATH)
        return
    if not os.path.exists(CLS_MODEL_PATH):
        print("Не знайдено класифікаційну модель:", CLS_MODEL_PATH)
        return
    if not os.path.exists(CLS_LABELS_PATH):
        print("Не знайдено файл класифікаційних міток:", CLS_LABELS_PATH)
        return

    # Завантаження детекційної моделі та міток
    print("Завантаження детекційної моделі...")
    detection_graph = load_detection_model(DET_MODEL_PATH)
    det_labels = load_labels(DET_LABELS_PATH)
    print("Детекційна модель завантажена.")

    # Завантаження класифікаційної моделі (TFLite) та міток
    print("Завантаження класифікаційної моделі...")
    cls_interpreter = Interpreter(model_path=CLS_MODEL_PATH)
    cls_interpreter.allocate_tensors()
    cls_labels = load_labels(CLS_LABELS_PATH)
    cls_input_details = cls_interpreter.get_input_details()
    cls_output_details = cls_interpreter.get_output_details()
    print("Класифікаційна модель завантажена.")

    # Отримання зображення: з камери або з файлу
    if args.camera:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Не вдалося відкрити камеру.")
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Не вдалося зчитати кадр з камери.")
                break
            boxes, scores, classes, num_detections = run_detection_inference(frame, detection_graph)
            output_frame = draw_detection_and_classification(frame.copy(), boxes, scores, classes, det_labels,
                                                             min_score_thresh=0.5, cls_interpreter=cls_interpreter,
                                                             cls_input_details=cls_input_details,
                                                             cls_output_details=cls_output_details,
                                                             cls_labels=cls_labels, num_detections=num_detections)
            cv2.imshow("Detection & Classification", output_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    elif args.image is not None:
        image = cv2.imread(args.image)
        if image is None:
            print("Не вдалося завантажити зображення:", args.image)
            return
        boxes, scores, classes, num_detections = run_detection_inference(image, detection_graph)
        output_image = draw_detection_and_classification(image.copy(), boxes, scores, classes, det_labels,
                                                         min_score_thresh=0.5, cls_interpreter=cls_interpreter,
                                                         cls_input_details=cls_input_details,
                                                         cls_output_details=cls_output_details,
                                                         cls_labels=cls_labels, num_detections=num_detections)
        cv2.imshow("Detection & Classification", output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Вкажіть --image <шлях> або --camera для захоплення з камери.")


if __name__ == '__main__':
    main()