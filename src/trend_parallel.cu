#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <map>
#include <cuda_runtime.h>

using namespace std;

//Map Pattern 
__global__ void computeTrend(const float* prices, int* trend, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n - 1) {
        trend[i] = (prices[i + 1] > prices[i]) ? 1 : 0;
    }
}

// อ่านไฟล์ CSV และจัดเก็บข้อมูลในแผนที่
void readStockCSV(const string& filename,
                  map<string, vector<string>>& allDates,
                  map<string, vector<float>>& allPrices) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: cannot open file " << filename << endl;
        return;
    }

    string line;
    bool headerSkipped = false;
    while (getline(file, line)) {
        if (!headerSkipped) { headerSkipped = true; continue; }
        if (line.empty()) continue;

        stringstream ss(line);
        string token;
        int col = 0;
        string indexName, date;
        float price = 0;
        bool valid = false;

        while (getline(ss, token, ',')) {
            if (col == 0) indexName = token;
            else if (col == 1) date = token;
            else if (col == 8) {
                try { price = stof(token); valid = true; }
                catch (...) { valid = false; }
                break;
            }
            col++;
        }

        if (valid) {
            allDates[indexName].push_back(date);
            allPrices[indexName].push_back(price);
        }
    }
    file.close();
}

int main() {
    string filename = "data/indexProcessed.csv";

    map<string, vector<string>> allDates;
    map<string, vector<float>> allPrices;

    auto ioStart = chrono::high_resolution_clock::now();
    readStockCSV(filename, allDates, allPrices);
    auto ioEnd = chrono::high_resolution_clock::now();
    cout << "CSV loaded in "
         << chrono::duration_cast<chrono::milliseconds>(ioEnd - ioStart).count()
         << " ms\n";
    cout << "Found " << allPrices.size() << " index groups.\n\n";

    for (auto it = allPrices.begin(); it != allPrices.end(); ++it) {
        const string& indexName = it->first;
        const vector<float>& prices = it->second;
        const vector<string>& dates = allDates[indexName];

        int N = prices.size();
        if (N < 2) continue;

        cout << "\n=== Index: " << indexName << " (" << N << " records) ===\n";

 
        float* d_prices;
        int* d_trend;
        cudaMalloc(&d_prices, N * sizeof(float));
        cudaMalloc(&d_trend, (N - 1) * sizeof(int));
        cudaMemcpy(d_prices, prices.data(), N * sizeof(float), cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (N - 1 + threads - 1) / threads;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        computeTrend<<<blocks, threads>>>(d_prices, d_trend, N);
        cudaGetLastError();
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float deviceMs = 0;
        cudaEventElapsedTime(&deviceMs, start, stop);

        vector<int> trend(N - 1);
        cudaMemcpy(trend.data(), d_trend, (N - 1) * sizeof(int), cudaMemcpyDeviceToHost);

        int longestRun = 0, currentRun = 0, bestEndIdx = 0;
        for (int i = 0; i < N - 1; ++i) {
            if (trend[i] == 1) {
                currentRun++;
                if (currentRun > longestRun) {
                    longestRun = currentRun;
                    bestEndIdx = i + 1;
                }
            } else currentRun = 0;
        }

        int bestStartIdx = bestEndIdx - longestRun;
        cout << "Device time (trend calc): " << deviceMs << " ms\n";
        cout << "Longest consecutive uptrend edges: " << longestRun << "\n";
        cout << "Longest consecutive uptrend days:  " << (longestRun + 1) << "\n";

        if (!dates.empty()) {
            cout << "Date Range: " << dates[bestStartIdx]
                 << " -> " << dates[bestEndIdx] << "\n";
        }

        cudaFree(d_prices);
        cudaFree(d_trend);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    return 0;
}