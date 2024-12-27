import cv2
import numpy as np
from ultralytics import YOLO

# Функция для вычисления расстояния между центрами боксов
def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Функция для обнаружения объектов (заглушка для YOLO)
def detect_objects(frame, model, confidence_threshold=0.5):
    """
    Заглушка для обнаружения объектов с помощью YOLO.
    Возвращает список обнаруженных объектов в формате [x1, y1, x2, y2, confidence, class_id].
    """
    # В реальной задаче замените это на вызов модели YOLO
    results = model.predict(frame, conf=confidence_threshold, device='cpu')
    detections = results[0].boxes.data.cpu().numpy()  # Получаем данные об обнаружениях
    detections = detections[detections[:, 4] >= confidence_threshold]  # Фильтруем по уверенности
    return detections

# Функция для слежения за объектами
def track_objects(detections, tracks, max_distance=10):
    """
    Слежение за объектами на основе расстояния между центрами боксов.
    """
    new_tracks = []
    for detection in detections:
        x1, y1, x2, y2, conf, class_id = detection
        center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])  # Центр бокса

        # Поиск ближайшего трека
        min_distance = max_distance
        match_track = None
        for track in tracks:
            track_center = np.array([(track['x1'] + track['x2']) / 2, (track['y1'] + track['y2']) / 2])
            distance = euclidean_distance(center, track_center)
            if distance < min_distance:
                min_distance = distance
                match_track = track

        if match_track:
            # Обновляем координаты трека
            match_track['x1'], match_track['y1'], match_track['x2'], match_track['y2'] = x1, y1, x2, y2
            new_tracks.append(match_track)
        else:
            # Создаем новый трек
            new_track = {'id': len(new_tracks) + 1, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
            new_tracks.append(new_track)

    return new_tracks

# Основная функция обработки видео
def process_video(input_path, output_path, model, confidence_threshold=0.45):
    """
    Обрабатывает видео, подсчитывая уникальных людей с использованием данных от YOLO.
    """
    try:
        # Открытие видеофайла
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видеофайл: {input_path}")

        # Получение информации о видео
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Настройка записи выходного видео (если требуется)
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        # Словарь для хранения уникальных людей
        unique_people = set()
        tracks = []  # Список активных треков

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            if(frame_count % 2 == 0):
                # Обнаружение объектов на кадре
                detections = detect_objects(frame, model, confidence_threshold)

                # Фильтрация обнаружений по классу "person" (класс 0)
                detections = detections[detections[:, 5] == 0]

                # Слежение за объектами
                tracks = track_objects(detections, tracks)

                # Учет уникальных людей
                for track in tracks:
                    track_id = track['id']
                    if track_id not in unique_people:
                        unique_people.add(track_id)

                    # Визуализация трека на кадре
                    cv2.rectangle(frame, (int(track['x1']), int(track['y1'])), (int(track['x2']), int(track['y2'])), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {track_id}", (int(track['x1']), int(track['y1']) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Отображение количества уникальных людей
                cv2.putText(frame, f"\n\nUnique People: {len(unique_people)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Отображение или сохранение кадра
                out.write(frame)
                out.write(frame)


        # Освобождение ресурсов
        cap.release()
        out.release()

        # Итоговый вывод
        print(f"Обработка видео завершена.")
        print(f"Обнаружено уникальных людей: {len(unique_people)}")
        if output_path:
            print(f"Результат сохранен в: {output_path}")

    except Exception as e:
        print(f"Произошла ошибка: {e}")

if __name__ == "__main__":
    # Загрузка модели YOLO
    model = YOLO('runs/detect/train2/weights/best.pt')  # Загружаем YOLO модель

    # Указание путей к видео
    input_video_path = 'videos/vid/1215.mp4'  # Замените на путь к вашему входному видео
    output_video_path = 'videos/res/output5.mp4'  # Замените на желаемый путь для сохранения выходного видео

    # Установка порога уверенности
    confidence = 0.45

    # Запуск обработки видео
    process_video(input_video_path, output_video_path, model, confidence)