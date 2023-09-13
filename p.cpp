//nasm -f elf64 matrix_multiplication.asm -o matrix_multiplication.o
//g++ -fopenmp -o output_file_name p.cpp -O3 -march=native -finline-functions -funroll-loops matrix_multiplication.o -m64 -no-pie
//./output_file_name

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <omp.h>

using namespace std;

extern "C" double matrixMultiplicationNASM(const double *H, const double *A, int k, int i, int j, int size);

//////////////////////////////////////Функция для создания полной матрицы Хаусхолдера////////////////////////////////////////////////
vector<vector<double>> createHouseholderMatrix(const vector<double>& v) {
    int n = v.size();
    vector<vector<double>> H(n, vector<double>(n, 0.0));

    double norm = 0.0;
    for (int i = 0; i < n; ++i) {
        norm += v[i] * v[i];
    }
    norm = sqrt(norm);

    vector<double> u(n);
    for (int i = 0; i < n; ++i) {
        u[i] = v[i] + (v[0] >= 0 ? norm : -norm);
    }

    double uuT = 0.0;
    for (int i = 0; i < n; ++i) {
        uuT += u[i] * u[i];
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            H[i][j] = -2 * u[i] * u[j] / uuT;
        }
        H[i][i] += 1.0;
    }

    return H;
}

////////////////////////////////// Функция для создания верхне-правой треугольной матрицы//////////////////////////////////////////////////////
vector<vector<double>> createUpperTriangularMatrix(int size) {
    vector<vector<double>> matrix(size, vector<double>(size, 0.0));

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(-1.0, 1.0);

    for (int i = 0; i < size; ++i) {
        for (int j = i; j < size; ++j) {
            matrix[i][j] = dis(gen);
        }
    }

    return matrix;
}

///////////////////////////////////// Функция для умножения матрицы на матрицу(без всего)///////////////////////////////////////////////////
vector<vector<double>> matrixMultiplication(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int n = A.size();
    int m = B[0].size();
    int p = B.size();

    vector<vector<double>> result(n, vector<double>(m, 0.0));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            for (int k = 0; k < p; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return result;
}

///////////////////////////////////////Функция для умножения матрицы на матрицу с OMP/////////////////////////////////////////////////
vector<vector<double>> matrixMultiplicationOMP(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int n = A.size();
    int m = B[0].size();
    int p = B.size();

    vector<vector<double>> result(n, vector<double>(m, 0.0));

	#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            for (int k = 0; k < p; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}
////////////////////Преобразование двумерного вектора в одномерный динамичекский массив/////////////////////////////////////
double* vector2DToDoublePtr(const std::vector<std::vector<double>>& vec2D) {
    // Получаем общее количество элементов
    size_t totalSize = 0;
    for (const auto& vec : vec2D) {
        totalSize += vec.size();
    }

    // Выделяем память для double*
    double* arr = new double[totalSize];

    // Копируем элементы в одномерный массив
    size_t index = 0;
    for (const auto& vec : vec2D) {
        for (const auto& value : vec) {
            arr[index] = value;
            index++;
        }
    }
    return arr;
}
///////////////////////////////////////Функция для умножения матрицы на матрицу с NASM///////////////////////////////////////////////
double *multiply(const double *H, const double *A, const int size) {
  auto *res = new double[size * size];
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      res[i * size + j] = 0;
      for (int k = 0; k < size; ++k) {
        double temp = 0;
        temp = matrixMultiplicationNASM(&H[0], &A[0], k, i, j, size);
        for (int l = k; l < size; ++l) {
        	temp += H[l * size + j] * A[k * size + l];
        }
        temp *= H[i * size + k];
        res[i * size + j] += temp;
      }
    }
  }
  return res;
}

///////////////////////////////////// Функция для вывода матрицы на экран///////////////////////////////////////////////////
void printMatrix(const vector<vector<double>>& matrix) {
    for (const auto& row : matrix) {
        for (const auto& element : row) {
            cout << element << " ";
        }
        cout << endl;
    }
}

////////////////////////////ГЛАВНАЯ ФУНКЦИЯ////////////////////////////////////////
int main() {
    int size;

    cout << "Введите размер матрицы: ";
    cin >> size;

    // Генерируем верхне-правую треугольную матрицу
    vector<vector<double>> A = createUpperTriangularMatrix(size);
    cout << "Верхне-правая треугольная матрица A создана" << endl;
//////////////////////////////////////////////////////////////////////////////////////
    // Генерируем случайные элементы для вектора
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(-1.0, 1.0);
    vector<double> v(size);
    for (int i = 0; i < size; ++i) {
        v[i] = dis(gen);
    }
    vector<vector<double>> H = createHouseholderMatrix(v);
    cout << "Householder matrix H создана" << endl;
    
//////////////////////////////////////////////////////////////////////////////////////	
	//Перемножение без расспаралеливания
	auto start = chrono::high_resolution_clock::now();
    vector<vector<double>> H_1 = matrixMultiplication(H, A);
    H_1 = matrixMultiplication(H_1, H);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout<<"Время выполнения перемножения без расспаралеливания:"<< elapsed.count()<<"сек."<<endl;
    
//////////////////////////////////////////////////////////////////////////////////////   
    //Перемножение с расспаралеливанием
	auto start1 = chrono::high_resolution_clock::now();
    vector<vector<double>> H_3 = matrixMultiplicationOMP(H, A);
    H_3 = matrixMultiplicationOMP(H_3, H);
    auto end1 = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed1 = end1 - start1;
    cout<<"Время выполнения перемножения c расспаралеливания:"<< elapsed1.count()<<"сек."<<endl;

//////////////////////////////////////////////////////////////////////////////////////  
	//Перемножение с NASM
    double* arr = vector2DToDoublePtr(H);
    double* arr1 = vector2DToDoublePtr(A);
    auto start2 = chrono::high_resolution_clock::now();
    auto *H_5 = multiply(arr, arr1, size);
    auto end2 = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed2 = end2 - start2;
    cout << "Время выполнения перемножения с использованием NASM:"<< elapsed2.count() << "сек." << endl;

///////////////////////////////////Перемножение матриц для проверки///////////////////////////////////////////////////     
    //Без
    vector<vector<double>> H_2 = matrixMultiplicationOMP(H, H_1);
    H_2 = matrixMultiplicationOMP(H_2, H);
    cout << "Q^T*B*Q (Проверка)" << endl;
//////////////////////////////////////////////////////////////////////////////////////
    //OMP
    vector<vector<double>> H_4 = matrixMultiplicationOMP(H, H_3);
    H_4 = matrixMultiplicationOMP(H_4, H);
    cout << "Q^T*B*Q OMP(Проверка)" << endl;

//////////////////////////Проверка////////////////////////////////////////////////////////////    
    bool matricesEqual = true;
for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
        if (abs(A[i][j] - H_2[i][j]) > 1e-9) {
            matricesEqual = false;
            break;
        }
    }
}
if (matricesEqual) {
    cout << "Матрицы A и H_2 равны." << endl;
} else {
    cout << "Матрицы A и H_2 не равны." << endl;
}
//////////////////////////////////////////////////////////////////////////    
    bool matricesEqual1 = true;
for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
        if (abs(A[i][j] - H_4[i][j]) > 1e-9) {
            matricesEqual1 = false;
            break;
        }
    }
}
if (matricesEqual1) {
    cout << "Матрицы A и H_4 равны." << endl;
} else {
    cout << "Матрицы A и H_4 не равны." << endl;
}

//////////////////////////////////////////////////////////////////////////    
    //H_1 - матрица полученная после перемножения без всего
    //H_2 - матрица полученная после перемножения для проверки H_1
    //H_3 - матрица полученная после перемножения с OMP
    //H_4 - матрица полученная после перемножения для проверки H_3
    //H_5 - матрица полученная после перемножения c NASM
    return 0;
}

