#include<Windows.h>
#include<iostream>
#include<fstream>
#include<vector>
#include<string>
#include<sstream>
#include<math.h>
using namespace std;

void string_to_num(string str, int row, int l, int** arr) 
{
    string s;
    int a;
    stringstream ss(str);

    while (ss >> s) 
	{
        stringstream ts;
        ts << s;
        ts >> a;
        arr[row][l - a - 1] = 1; 
    }
}


int get_first_1(int* arr, int size) 
{
    for (int i = 0; i < size; i++) 
	{
        if (arr[i] == 1)
            return size - 1 - i;
        else
            continue;
    }
    return -1;
}

int _exist(int** E, int* Ed, int row, int line) 
{
    for (int i = 0; i < row; i++) 
	{
        if (get_first_1(E[i], line) == get_first_1(Ed, line)) 
            return i; 
    }
    return -1;
}

void special_Gauss(int** E, int** Ed, int row, int rowd, int line) {
    int count = row - rowd;
    long long head, tail, freq;
    double sum_time = 0.0;
    for (int i = 0; i < rowd; i++) 
	{ 
        while (get_first_1(Ed[i], line) != -1) 
		{ 
            int exist_or_not = _exist(E, Ed[i], row, line);
            if (exist_or_not != -1) 
			{ 
                QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
                QueryPerformanceCounter((LARGE_INTEGER*)&head);
                int k;
                for (k = 0; k < line; k++) 
				{
                    Ed[i][k] = Ed[i][k] ^ E[exist_or_not][k];
                }
                QueryPerformanceCounter((LARGE_INTEGER*)&tail);
                sum_time += (tail - head) * 1000.0 / freq;
            }
            else 
			{ 
                for (int k = 0; k < line; k++) 
				{
                    E[count][k] = Ed[i][k];
                }
                count++;
                break; 
            }
        }
    }
    cout << "Time:" << sum_time << "ms" << endl; 
}

int main() {
    ifstream eliminate;
    ifstream eliminated;
    ifstream data;
    ofstream result;
    int row, line;
    int rowd, lined;
    data.open("D:\\codeblockS\\test\\readme.txt", ios::in);
    data >> line;
    data >> row;
    data >> rowd;
    lined = line;
    row += rowd;
    int** E = new int* [row];
    for (int i = 0; i < row; i++)
        E[i] = new int[line];
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < line; j++) 
		{
            E[i][j] = 0;
        }
    }
    int** Ed = new int* [rowd];
    for (int i = 0; i < rowd; i++)
        Ed[i] = new int[lined];
    for (int i = 0; i < rowd; i++) 
	{
        for (int j = 0; j < lined; j++) 
		{
            Ed[i][j] = 0;
        }
    }

    eliminate.open("D:\\codeblockS\\test\\Eliminant.txt", ios::in);
    if (!eliminate.is_open()) 
	{
        cout << "Failed to open the file.£¨The read-in file doesn't exist£©" << endl;
        return 1;
    }
    vector<string> elte;
    string temp1;
    while (getline(eliminate, temp1))
        elte.push_back(temp1);
    eliminate.close();
    for (int i = 0; i < elte.size(); i++)
        string_to_num(elte[i], i, line, E);
    eliminated.open("D:\\codeblockS\\test\\Eliminated_rows.txt", ios::in);
    if (!eliminated.is_open()) 
	{
        cout << "Failed to open the file.£¨The read-in file doesn't exist£©'£©" << endl;
        return 1;
    }
    vector<string> elted;
    string temp2;
    while (getline(eliminated, temp2))
        elted.push_back(temp2);
    eliminated.close();
    for (int i = 0; i < elted.size(); i++)
        string_to_num(elted[i], i, lined, Ed);

    special_Gauss(E, Ed, row, rowd, line); 
    result.open("D:\\codeblockS\\test\\Elimination_result.txt", ios::out);
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < line; j++)
        {
            result << E[i][j];
        }
        result << endl;
    }
    for (int i = 0; i < row; i++)
        delete[] E[i];
    delete[] E;
    for (int i = 0; i < rowd; i++)
        delete[] Ed[i];
    delete[]Ed;
    return 0;
}
