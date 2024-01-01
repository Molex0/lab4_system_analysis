# Список для збереження рядків з кожного файлу
lines_files = []

# Відкриття та зчитування рядків з кожного файлу
for i in range(1, 6):
    filename = f'argF{i}_Warn.txt'
    with open(filename, 'r') as file:
        lines = file.readlines()
        lines_files.append(lines)

# Об'єднання рядків з усіх файлів разом
merged_lines = ''
for lines in zip(*lines_files):
    merged_lines += ' '.join(line.strip() for line in lines) + '\n'

# Запис об'єднаних рядків у новий файл
with open('Warn.txt', 'w') as merged_file:
    merged_file.write(merged_lines)