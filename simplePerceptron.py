# импортируем модуль numpy для работы с многомерными массивами
import numpy as np


# определяем функцию сигмоид для реализации функции активатора
def sigmoid (x):
    # из numpy вызываем метод exp для вычисления экспоненты всех
    # элементов входного массива
    return 1 / (1 + np.exp(-x))

# входящие тренировочные значения
test_inputs = np.array([[0, 0, 1],
                        [0, 1, 1],
                        [1, 1, 1],
                        [1, 1, 0,]])

# ожидаемые выходные данные
test_outputs = np.array([[0, 0, 1, 1]]).T

# инициализация весов
np.random.seed(1)

# получаем инициализированные веса массивом 3 на 1
synaptic_weights = 2 * np.random.random((3, 1)) - 1

print("Случайные инициализирующие веса: ")
print(synaptic_weights)

# так как веса заданы явно, скрипт не пригоден для использования
# меняем ситуацию. Используем метод обучения "обратное распространение"
for p in range(20000):
    input_layer = test_inputs
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))

    err = test_outputs - outputs
    adj = np.dot(input_layer.T, err * (outputs * (1 - outputs)))

    synaptic_weights += adj

print("Веса после обучения:")
print (synaptic_weights)

print("Результат после обучения:")
print(outputs)

# Простейший пример нейронки Перцептрон обучена
