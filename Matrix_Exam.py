import argparse
from functools import wraps
import logging
import random
import sys

# Практическое задание Урок 11 Задание 4: Задача о матричных операциях


def log(func):
    """
    Декоратор для записи логов.

    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f'Call function {func}')
        try:
            func_origin = func(*args, **kwargs)
            logging.info(f'End function {func}')
            return func_origin
        except Exception as error:
            logging.exception(error)
            raise error

    return wrapper


class Matrix:
    """
    Класс, представляющий матрицу.

    Атрибуты:
    - rows (int): количество строк в матрице
    - cols (int): количество столбцов в матрице
    - data (list): двумерный список, содержащий элементы матрицы

    Методы:
    - gen_matrix(): генерирует матрицу
    - __str__(): возвращает строковое представление матрицы
    - __repr__(): возвращает строковое представление матрицы, которое может быть использовано для создания нового объекта
    - __eq__(other): определяет операцию "равно" для двух матриц
    - __add__(other): определяет операцию сложения двух матриц
    - __mul__(other): определяет операцию умножения двух матриц
    """

    @classmethod
    @log
    def gen_matrix(cls, n: int, m: int):
        """
        Возвращает сгенерированную матрицу.

        Возвращает:
        - Matrix: матрица
        """
        matrix = cls(n, m)
        matrix.data = [[random.randint(1, 10) for j in range(matrix.cols)]
                       for i in range(matrix.rows)]
        return matrix

    @log
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.data = [[0 for j in range(cols)] for i in range(rows)]

    @log
    def __str__(self):
        """
        Возвращает строковое представление матрицы.

        Возвращает:
        - str: строковое представление матрицы
        """
        matrix_str = '\n'.join([' '.join([str(self.data[i][j])
                               for j in range(self.cols)])
                               for i in range(self.rows)])
        return matrix_str

    @log
    def __repr__(self):
        """
        Возвращает строковое представление матрицы, которое может быть использовано для создания нового объекта.

        Возвращает:
        - str: строковое представление матрицы
        """
        return f"Matrix({self.rows}, {self.cols})"

    @log
    def __eq__(self, other):
        """
        Определяет операцию "равно" для двух матриц.

        Аргументы:
        - other (Matrix): вторая матрица

        Возвращает:
        - bool: True, если матрицы равны, иначе False
        """
        if self.rows != other.rows or self.cols != other.cols:
            return False
        for i in range(self.rows):
            for j in range(self.cols):
                if self.data[i][j] != other.data[i][j]:
                    return False
        return True

    @log
    def __add__(self, other):
        """
        Определяет операцию сложения двух матриц.

        Аргументы:
        - other (Matrix): вторая матрица

        Возвращает:
        - Matrix: новая матрица, полученная путем сложения двух исходных матриц
        """
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Матрицы должны иметь одинаковые размеры")
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] = self.data[i][j] + other.data[i][j]
        return result

    @log
    def __mul__(self, other):
        """
        Определяет операцию умножения двух матриц.

        Аргументы:
        - other (Matrix): вторая матрица

        Возвращает:
        - Matrix: новая матрица, полученная путем умножения двух исходных матриц
        """
        if self.cols != other.rows:
            raise ValueError(
                "Количество столбцов первой матрицы должно быть равно количеству строк второй матрицы")
        result = Matrix(self.rows, other.cols)
        for i in range(self.rows):
            for j in range(other.cols):
                for k in range(self.cols):
                    result.data[i][j] += self.data[i][k] * other.data[k][j]
        return result


logging.basicConfig(level=logging.INFO, filename="runtime.log",
                    filemode="w", format="%(asctime)s %(levelname)s %(message)s")

print(sys.argv)

# Создаем матрицы
matrix1: Matrix
matrix2: Matrix

if len(sys.argv) > 1:
    parser = argparse.ArgumentParser(description='Matrix')
    parser.add_argument("--n1", type=int, required=True)
    parser.add_argument("--m1", type=int, required=True)
    parser.add_argument("--n2", type=int, required=True)
    parser.add_argument("--m2", type=int, required=True)

    args = parser.parse_args()

    matrix1 = Matrix.gen_matrix(args.n1, args.m1)
    matrix2 = Matrix.gen_matrix(args.n2, args.m2)
else:
    matrix1 = Matrix(2, 3)
    matrix1.data = [[1, 2, 3], [4, 5, 6]]
    matrix2 = Matrix(2, 3)
    matrix2.data = [[7, 8, 9], [10, 11, 12]]

# Выводим матрицы
print(matrix1)

print(matrix2)

# Сравниваем матрицы
print(matrix1 == matrix2)

# Выполняем операцию сложения матриц
matrix_sum = matrix1 + matrix2
print(matrix_sum)

# Выполняем операцию умножения матриц
matrix3 = Matrix(3, 2)
matrix3.data = [[1, 2], [3, 4], [5, 6]]

matrix4 = Matrix(2, 2)
matrix4.data = [[7, 8], [9, 10]]

result = matrix3 * matrix4
print(result)
