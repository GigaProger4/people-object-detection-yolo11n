import os

def find_matching_txt(image_path):
    # Извлечение имени файла без расширения
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # Формирование пути к соответствующему файлу txt
    txt_path = os.path.join(os.path.dirname('C:/Users/илья/PycharmProjects/pythonProject3/neuro_data/cat/labels/train'), image_name + '.txt')

    # Проверка существования файла txt
    if os.path.exists(image_path):
        print(111)
    else:
        print(f"Файл {txt_path} не найден.")

def process_images_in_directory(directory):
    for filename in os.listdir(directory):
        image_path = os.path.join(directory, filename)
        find_matching_txt(image_path)

# Пример использования
image_path = "C:/Users/илья/PycharmProjects/pythonProject3/neuro_data/cat/images/train/"
process_images_in_directory(image_path)
import os

def find_txt_files_by_content(directory, target_content):
    matching_files = []

    for filename in os.listdir(directory):
        if filename.lower().endswith('.txt'):
            file_path = os.path.join(directory, filename)

            with open(file_path, 'r') as file:
                content = file.read()

                if target_content in content:
                    matching_files.append(filename)

    return matching_files

# Пример использования
image_directory = "C:/Users/илья/PycharmProjects/pythonProject3/neuro_data/cat/labels/train/"
target_content = "-1 -1 -1 -1"

matching_files = find_txt_files_by_content(image_directory, target_content)

if matching_files:
    print(f"Найдены следующие файлы, содержащие строку '{target_content}':")
    for filename in matching_files:
        print(f"- {filename}")
else:
    print(f"Файлы, содержащие строку '{target_content}', не найдены.")