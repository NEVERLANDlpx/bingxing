#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <map>
#include <vector>
#include <windows.h>
#include <tmmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <smmintrin.h>
#include <nmmintrin.h>
#include <immintrin.h>
#include <mpi.h>
#include <omp.h>

using namespace std;

#define NUM_THREADS 8

const int maxsize = 3000;
const int maxrow = 40000;
const int numBasis = 40000;
const int bitset_size = 96000; // 3000 * 32

vector<int> tmpAns;
long long head, tail, freq;

map<int, vector<pair<int, int>>> ans;

fstream RowFile("被消元行.txt", ios::in | ios::out);
fstream BasisFile("消元子.txt", ios::in | ios::out);
ofstream out_mpi("消元结果(MPI).txt");

vector<vector<pair<int, int>>> gRows(maxrow);
vector<vector<pair<int, int>>> gBasis(numBasis);
vector<vector<pair<int, int>>> answers(maxrow);
map<int, int> firstToRow;

int ifBasis[numBasis] = {0};
int ifDone[maxrow] = {0};

void reset() {
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
        getline(BasisFile, tmp);
        stringstream s(tmp);
        int pos;
        while (s >> pos) {
            gBasis[i].push_back({i, pos});
        }
    }
    return numBasis;
}

int readRowsFrom(int pos) {
    RowFile.clear();
    RowFile.seekg(0, ios::beg);
    string line;
    for (int i = 0; i < pos; i++) {
        getline(RowFile, line);
    }
    for (int i = 0; i < maxrow; i++) {
        if (getline(RowFile, line).eof()) {
            cout << "读取被消元行 " << i << " 行" << endl;
            return i;
        }
        stringstream s(line);
        int tmp;
        while (s >> tmp) {
            gRows[i].push_back({i, tmp});
        }
    }
    cout << "read max rows" << endl;
    return -1;
}

int findfirst(int row) {
    if (gRows[row].empty()) return -1;
    return gRows[row].back().second;
}

int _findfirst(int row) {
    if (answers[row].empty()) return -1;
    return answers[row].back().second;
}

void writeResult(ofstream& out) {
    for (auto it = ans.rbegin(); it != ans.rend(); it++) {
        vector<pair<int, int>>& result = it->second;
        for (auto& p : result) {
            out << p.second << " ";
        }
        out << endl;
    }
}

void writeResult_MPI(ofstream& out) {
    for (int j = 0; j < maxrow; j++) {
        for (auto& p : answers[j]) {
            out << p.second << " ";
        }
        out << endl;
    }
}

void AVX_GE() {
    int begin = 0;
    int flag = readRowsFrom(begin);
    int num = (flag == -1) ? maxrow : flag;
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    for (int i = 0; i < num; i++) {
        while (findfirst(i) != -1) {
            int first = findfirst(i);
            if (ifBasis[first] == 1) {
                for (auto& p : gBasis[first]) {
                    auto it = find_if(gRows[i].begin(), gRows[i].end(), [&](pair<int, int>& q) {
                        return q.second == p.second;
                    });
                    if (it != gRows[i].end()) {
                        gRows[i].erase(it);
                    } else {
                        gRows[i].push_back(p);
                    }
                }
            } else {
                gBasis[first] = gRows[i];
                ifBasis[first] = 1;
                ans.insert(pair<int, vector<pair<int, int>>>(first, gBasis[first]));
                break;
            }
        }
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "AVX time:" << (tail - head) * 1000 / freq << "ms" << endl;
}

void GE_MPI_AVX_omp(int argc, char* argv[]) {
    int flag;
    double start_time = 0;
    double end_time = 0;
    MPI_Init(&argc, &argv);
    int total = 0;
    int rank = 0;
    int i = 0;
    int j = 0;
    int begin = 0, end = 0;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &total);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        flag = readRowsFrom(0);
        num = (flag == -1) ? maxrow : flag;
        begin = rank * num / total;
        end = (rank == total - 1) ? num : (rank + 1) * (num / total);
        for (i = 1; i < total; i++) {
            MPI_Send(&num, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            int b = i * (num / total);
            int e = (i == total - 1) ? num : (i + 1) * (num / total);
            for (j = b; j < e; j++) {
                MPI_Send(gRows[j].data(), gRows[j].size() * sizeof(pair<int, int>), MPI_BYTE, i, 1, MPI_COMM_WORLD);
            }
        }
    } else {
        MPI_Recv(&num, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        begin = rank * (num / total);
        end = (rank == total - 1) ? num : (rank + 1) * (num / total);
        for (i = begin; i < end; i++) {
            vector<pair<int, int>> tmp(bitset_size / 8);
            MPI_Recv(tmp.data(), bitset_size / 8, MPI_BYTE, 0, 1, MPI_COMM_WORLD, &status);
            gRows[i] = tmp;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
#pragma omp parallel num_threads(NUM_THREADS), private(i, j)
#pragma omp for ordered schedule(guided)
    for (i = begin; i < end; i++) {
        int first = findfirst(i);
        while (first != -1) {
            if (ifBasis[first] == 1) {
                for (auto& p : gBasis[first]) {
                    auto it = find_if(gRows[i].begin(), gRows[i].end(), [&](pair<int, int>& q) {
                        return q.second == p.second;
                    });
                    if (it != gRows[i].end()) {
                        gRows[i].erase(it);
                    } else {
                        gRows[i].push_back(p);
                    }
                }
                first = findfirst(i);
            } else {
#pragma omp ordered
                if (rank == 0) {
                    while (ifBasis[first] == 1) {
                        for (auto& p : gBasis[first]) {
                            auto it = find_if(gRows[i].begin(), gRows[i].end(), [&](pair<int, int>& q) {
                                return q.second == p.second;
                            });
                            if (it != gRows[i].end()) {
                                gRows[i].erase(it);
                            } else {
                                gRows[i].push_back(p);
                            }
                        }
                        first = findfirst(i);
                    }
                    if (first != -1) {
                        gBasis[first] = gRows[i];
                        answers[i] = gRows[i];
                        ifBasis[first] = 1;
                    }
                }
                first = -1;
            }
        }
    }

    for (i = 0; i < rank; i++) {
        int b = i * (num / total);
        int e = b + num / total;
        for (j = b; j < e; j++) {
            vector<pair<int, int>> tmp(bitset_size / 8);
            MPI_Recv(tmp.data(), bitset_size / 8, MPI_BYTE, i, 2, MPI_COMM_WORLD, &status);
            answers[j] = tmp;
            int first = _findfirst(j);
            firstToRow.insert(pair<int, int>(first, j));
        }
#pragma omp for schedule(guided)
        for (j = begin; j < end; j++) {
            int first = findfirst(j);
            while ((firstToRow.find(first) != firstToRow.end() || ifBasis[first] == 1) && first != -1) {
                if (firstToRow.find(first) != firstToRow.end()) {
                    int row = firstToRow.find(first)->second;
                    for (auto& p : answers[row]) {
                        auto it = find_if(gRows[j].begin(), gRows[j].end(), [&](pair<int, int>& q) {
                            return q.second == p.second;
                        });
                        if (it != gRows[j].end()) {
                            gRows[j].erase(it);
                        } else {
                            gRows[j].push_back(p);
                        }
                    }
                    first = findfirst(i);
                }
                if (ifBasis[first] == 1) {
                    for (auto& p : gBasis[first]) {
                        auto it = find_if(gRows[j].begin(), gRows[j].end(), [&](pair<int, int>& q) {
                            return q.second == p.second;
                        });
                        if (it != gRows[j].end()) {
                            gRows[j].erase(it);
                        } else {
                            gRows[j].push_back(p);
                        }
                    }
                    first = findfirst(i);
                }
            }
        }
    }

    if (rank != 0) {
        for (i = begin; i < end; i++) {
            int first = findfirst(i);
            if (first != -1) {
                while ((firstToRow.find(first) != firstToRow.end() || ifBasis[first] == 1) && first != -1) {
                    if (firstToRow.find(first) != firstToRow.end()) {
                        int row = firstToRow.find(first)->second;
                        for (auto& p : answers[row]) {
                            auto it = find_if(gRows[i].begin(), gRows[i].end(), [&](pair<int, int>& q) {
                                return q.second == p.second;
                            });
                            if (it != gRows[i].end()) {
                                gRows[i].erase(it);
                            } else {
                                gRows[i].push_back(p);
                            }
                        }
                        first = findfirst(i);
                    }
                    if (ifBasis[first] == 1) {
                        for (auto& p : gBasis[first]) {
                            auto it = find_if(gRows[i].begin(), gRows[i].end(), [&](pair<int, int>& q) {
                                return q.second == p.second;
                            });
                            if (it != gRows[i].end()) {
                                gRows[i].erase(it);
                            } else {
                                gRows[i].push_back(p);
                            }
                        }
                        first = findfirst(i);
                    }
                }
                gBasis[first] = gRows[i];
                answers[i] = gRows[i];
                ifBasis[first] = 1;
            }
        }
    }

    for (i = rank + 1; i < total; i++) {
        for (j = begin; j < end; j++) {
            MPI_Send(answers[j].data(), answers[j].size() * sizeof(pair<int, int>), MPI_BYTE, i, 2, MPI_COMM_WORLD);
        }
    }

    if (rank == total - 1) {
        end_time = MPI_Wtime();
        cout << "MPI+omp+AVX优化版本耗时： " << 1000 * (end_time - start_time) << "ms" << endl;
        writeResult_MPI(out_mpi);
        out_mpi.close();
    }
    MPI_Finalize();
}

int main(int argc, char* argv[]) {
    ofstream out("消元结果.txt");
    ofstream out1("消元结果(AVX).txt");
    ofstream out2("消元结果(GE_lock).txt");
    ofstream out3("消元结果(AVX_lock).txt");
    ofstream out4("消元结果(GE_omp).txt");
    ofstream out5("消元结果(AVX_omp).txt");
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

    readBasis();
    GE_MPI_AVX_omp(argc, argv);
    cout << "done!" << endl;

    out.close();
    out1.close();
    out2.close();
    out3.close();
    out4.close();
    out5.close();
}

