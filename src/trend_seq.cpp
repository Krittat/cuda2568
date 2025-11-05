#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <map>
#include <iomanip> 

using namespace std;

struct PriceRecord {
    string indexName;
    string date;
    float closePrice;
};

// อ่านข้อมูล CSV
vector<PriceRecord> readPricesFromCSV(const string& filename) {
    vector<PriceRecord> records;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: cannot open file " << filename << endl;
        return records;
    }

    string line;
    bool headerSkipped = false;

    while (getline(file, line)) {
        if (!headerSkipped) { headerSkipped = true; continue; }
        if (line.empty()) continue;

        stringstream ss(line);
        string token;
        int columnIndex = 0;
        string indexName, date;
        float closePrice = 0.0f;
        bool valid = false;

        while (getline(ss, token, ',')) {
            if (columnIndex == 0) indexName = token;
            else if (columnIndex == 1) date = token;
            else if (columnIndex == 8) {
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

        if (valid)
            records.push_back({indexName, date, closePrice});
    }

    file.close();
    return records;
}

// ตรวจจับแนวโน้มราคาหุ้น
vector<int> detectTrendSequential(const vector<PriceRecord>& data) {
    int n = data.size();
    vector<int> trend(n - 1, 0);
    for (int i = 0; i < n - 1; ++i) {
        if (data[i + 1].closePrice > data[i].closePrice) trend[i] = 1;
        else if (data[i + 1].closePrice < data[i].closePrice) trend[i] = -1;
        else trend[i] = 0;
    }
    return trend;
}

int main() {
    string filename = "data/indexProcessed.csv";
    vector<PriceRecord> allData = readPricesFromCSV(filename);

    if (allData.empty()) {
        cerr << "No data found.\n";
        return 1;
    }

    map<string, vector<PriceRecord>> groupedData;
    for (auto& rec : allData) {
        groupedData[rec.indexName].push_back(rec);
    }

    cout << fixed << setprecision(3);
    cout << "Loaded " << allData.size() << " total records.\n";
    cout << "Found " << groupedData.size() << " indices.\n\n";

    for (auto& [indexName, data] : groupedData) {
        if (data.size() < 2) continue;

        auto start = chrono::high_resolution_clock::now();
        vector<int> trend = detectTrendSequential(data);
        auto end = chrono::high_resolution_clock::now();

        double duration_ms = chrono::duration<double, milli>(end - start).count();

        int longestRun = 0, currentRun = 0;
        int longestStartIdx = 0, currentStartIdx = 0;

        for (int i = 0; i < trend.size(); ++i) {
            if (trend[i] == 1) {
                if (currentRun == 0) currentStartIdx = i;
                currentRun++;
                if (currentRun > longestRun) {
                    longestRun = currentRun;
                    longestStartIdx = currentStartIdx;
                }
            } else currentRun = 0;
        }

        cout << "=== Index: " << indexName << " (" << data.size() << " records) === "<< endl;
        cout << "Detection time: " << duration_ms << " ms\n";
        cout << "Longest consecutive increasing edges: " << longestRun << endl;
        cout << "Longest consecutive increasing days: " << (longestRun + 1) << endl;

        if (longestRun > 0) {
            cout << "Date range: " 
                 << data[longestStartIdx].date << "->"
                 << data[longestStartIdx + longestRun].date << endl;
            // cout << "   Price range: " 
            //      << data[longestStartIdx].closePrice << "->"
            //      << data[longestStartIdx + longestRun].closePrice << endl;
        }
        cout << "\n";
    }

    return 0;
}
