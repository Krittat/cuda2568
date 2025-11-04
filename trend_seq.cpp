#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>

using namespace std;

// อ่านข้อมูลราคาหุ้นจากไฟล์ CSV (ใช้เฉพาะคอลัมน์ Close ที่ index = 5)
vector<float> readPricesFromCSV(const string& filename) {
    vector<float> prices;
    
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: cannot open file " << filename << endl;
        return prices;
    }

    string line;
    bool headerSkipped = false;

    while (getline(file, line)) {
        if (!headerSkipped) { headerSkipped = true; continue; }
        if (line.empty()) continue; 

        stringstream ss(line);
        string token;
        int columnIndex = 0;
        float closePrice = 0.0f;
        bool valid = false;

        while (getline(ss, token, ',')) {
            if (columnIndex == 5) {
                try {
                    closePrice = stof(token);
                    valid = true;
                } catch (...) {
                    valid = false;
                }
                break;
            }
            columnIndex++;
        }

        if (valid) prices.push_back(closePrice); 
    }

    file.close();
    return prices;
}

// ฟังก์ชันตรวจจับแนวโน้มราคาหุ้นแบบ Sequential
vector<int> detectTrendSequential(const vector<float>& prices) {
    int n = prices.size();
    vector<int> trend(n - 1, 0);

    for (int i = 0; i < n - 1; ++i) {
        if (prices[i + 1] > prices[i]) trend[i] = 1;
        else if (prices[i + 1] < prices[i]) trend[i] = -1;
        else trend[i] = 0;
    }
    return trend;
}

int main() {
    string filename = "data/indexData.csv"; 
    vector<float> prices = readPricesFromCSV(filename);
    cout << "DEBUG: prices.size() = " << prices.size() << endl;

    if (prices.size() < 2) {
        cerr << "Error: not enough data points in " << filename << endl;
        return 1;
    }

    cout << "Loaded " << prices.size() << " price records.\n";

    auto start = chrono::high_resolution_clock::now();
    vector<int> trend = detectTrendSequential(prices);
    auto end = chrono::high_resolution_clock::now();

    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Sequential detection time: " << duration.count() << " ms\n";

    // แสดงผลบางส่วนเพื่อเช็คความถูกต้อง
    cout << "Sample trend result:\n";
    for (int i = 0; i < min(20, (int)trend.size()); ++i) {
        cout << trend[i] << " ";
    }
    cout << "\n";

    return 0;
}
