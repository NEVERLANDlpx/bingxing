#include <CL/sycl.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <chrono>
using namespace std;
using namespace sycl;

#define NUM_THREADS 8
const int maxsize = 3000;
const int maxrow = 40000;
const int numBasis = 40000;
int num;

vector<int> tmpAns;
long long head, tail, freq;
map<int, int*> ans;

fstream RowFile("被消元行.txt", ios::in | ios::out);
fstream BasisFile("消元子.txt", ios::in | ios::out);

ofstream out_mpi("消元结果(MPI).txt");

int gRows[maxrow][maxsize];
int gBasis[numBasis][maxsize];
int answers[maxrow][maxsize];
map<int, int> firstToRow;

int ifBasis[numBasis] = { 0 };
int ifDone[maxrow] = { 0 };

void reset() {
    memset(gRows, 0, sizeof(gRows));
    memset(gBasis, 0, sizeof(gBasis));
    memset(ifBasis, 0, sizeof(ifBasis));
    RowFile.close();
    BasisFile.close();
    RowFile.open("被消元行.txt", ios::in | ios::out);
    BasisFile.open("消元子.txt", ios::in | ios::out);
    ans.clear();
}

int readBasis() {
    for (int i = 0; i < numBasis; i++) {
        if (BasisFile.eof()) {
            cout << "读取消元子" << i - 1 << "行" << endl;
            return i - 1;
        }
        string tmp;
        bool flag = false;
        int row = 0;
        getline(BasisFile, tmp);
        stringstream s(tmp);
        int pos;
        while (s >> pos) {
            if (!flag) {
                row = pos;
                flag = true;
                ifBasis[row] = 1;
            }
            int index = pos / 32;
            int offset = pos % 32;
            gBasis[row][index] = gBasis[row][index] | (1 << offset);
        }
        flag = false;
        row = 0;
    }
}

int readRowsFrom(int pos) {
    if (RowFile.is_open())
        RowFile.close();
    RowFile.open("被消元行.txt", ios::in | ios::out);
    memset(gRows, 0, sizeof(gRows));
    string line;
    for (int i = 0; i < pos; i++) {
        getline(RowFile, line);
    }
    for (int i = pos; i < pos + maxrow; i++) {
        int tmp;
        getline(RowFile, line);
        if (line.empty()) {
            cout << "读取被消元行 " << i << " 行" << endl;
            return i;
        }
        bool flag = false;
        stringstream s(line);
        while (s >> tmp) {
            int index = tmp / 32;
            int offset = tmp % 32;
            gRows[i - pos][index] = gRows[i - pos][index] | (1 << offset);
            flag = true;
        }
    }
    cout << "read max rows" << endl;
    return -1;
}

int findfirst(int row) {
    int first;
    for (int i = maxsize - 1; i >= 0; i--) {
        if (gRows[row][i] == 0)
            continue;
        else {
            int pos = i * 32;
            int offset = 0;
            for (int k = 31; k >= 0; k--) {
                if (gRows[row][i] & (1 << k)) {
                    offset = k;
                    break;
                }
            }
            first = pos + offset;
            return first;
        }
    }
    return -1;
}

int _findfirst(int row) {
    int first;
    for (int i = maxsize - 1; i >= 0; i--) {
        if (answers[row][i] == 0)
            continue;
        else {
            int pos = i * 32;
            int offset = 0;
            for (int k = 31; k >= 0; k--) {
                if (answers[row][i] & (1 << k)) {
                    offset = k;
                    break;
                }
            }
            first = pos + offset;
            return first;
        }
    }
    return -1;
}

void writeResult(ofstream &out) {
    for (auto it = ans.rbegin(); it != ans.rend(); it++) {
        int* result = it->second;
        int max = it->first / 32 + 1;
        for (int i = max; i >= 0; i--) {
            if (result[i] == 0)
                continue;
            int pos = i * 32;
            for (int k = 31; k >= 0; k--) {
                if (result[i] & (1 << k)) {
                    out << k + pos << " ";
                }
            }
        }
        out << endl;
    }
}

void GE_GPU(int argc, char* argv[]) {
    int flag;
    double start_time = 0;
    double end_time = 0;

    // Set up SYCL queue with default device selector
    queue q;

    int* d_rows = malloc_device<int>(maxrow * maxsize, q);
    int* d_basis = malloc_device<int>(numBasis * maxsize, q);
    int* d_ifBasis = malloc_device<int>(numBasis, q);
    int* d_answers = malloc_device<int>(maxrow * maxsize, q);

    flag = readRowsFrom(0);
    num = (flag == -1) ? maxrow : flag;

    q.memcpy(d_rows, gRows, maxrow * maxsize * sizeof(int)).wait();
    q.memcpy(d_basis, gBasis, numBasis * maxsize * sizeof(int)).wait();
    q.memcpy(d_ifBasis, ifBasis, numBasis * sizeof(int)).wait();
    q.memcpy(d_answers, answers, maxrow * maxsize * sizeof(int)).wait();

    auto start = chrono::high_resolution_clock::now();

    q.parallel_for(range<1>(num), [=](id<1> idx) {
        int i = idx[0];
        int row_start = i * maxsize;
        while (true) {
            int first = -1;
            for (int j = maxsize - 1; j >= 0; j--) {
                if (d_rows[row_start + j] != 0) {
                    int pos = j * 32;
                    for (int k = 31; k >= 0; k--) {
                        if (d_rows[row_start + j] & (1 << k)) {
                            first = pos + k;
                            break;
                        }
                    }
                    break;
                }
            }
            if (first == -1) break;

            if (d_ifBasis[first] == 1) {
                for (int j = 0; j < maxsize; j++) {
                    d_rows[row_start + j] ^= d_basis[first * maxsize + j];
                }
            } else {
                for (int j = 0; j < maxsize; j++) {
                    d_basis[first * maxsize + j] = d_rows[row_start + j];
                    d_answers[row_start + j] = d_rows[row_start + j];
                }
                d_ifBasis[first] = 1;
                break;
            }
        }
    }).wait();

    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    cout << "GPU优化版本耗时： " << duration.count() << "ms" << endl;

    q.memcpy(gRows, d_rows, maxrow * maxsize * sizeof(int)).wait();
    q.memcpy(gBasis, d_basis, numBasis * maxsize * sizeof(int)).wait();
    q.memcpy(ifBasis, d_ifBasis, numBasis * sizeof(int)).wait();
    q.memcpy(answers, d_answers, maxrow * maxsize * sizeof(int)).wait();

    free(d_rows, q);
    free(d_basis, q);
    free(d_ifBasis, q);
    free(d_answers, q);

    for (int i = 0; i < num; i++) {
        if (findfirst(i) != -1) {
            ans.insert(pair<int, int*>(findfirst(i), gBasis[findfirst(i)]));
        }
    }
}

int main(int argc, char* argv[]) {
    ofstream out("消元结果.txt");

    readBasis();
    GE_GPU(argc, argv);
    writeResult(out);

    out.close();
    return 0;
}

