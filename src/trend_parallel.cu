#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <map>
#include <cuda_runtime.h>
#include <algorithm>

using namespace std;

#define BLOCK_SIZE 256

//Mapping pattern - คำนวณ trend (uptrend = 1, downtrend = 0)
__global__ void computeTrend(const float* prices, int* trend, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n - 1)
        trend[i] = (prices[i + 1] > prices[i]) ? 1 : 0;
}

// K
__global__ void computeRunLengths(const int* trend, int* run_lengths, int n_edges) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_edges) {
        if (trend[i] == 1) {
            int len = 1;
            for (int j = i - 1; j >= 0 && trend[j] == 1; --j)
                len++;
            run_lengths[i] = len;
        } else {
            run_lengths[i] = 0;
        }
    }
}

// Reduction หาค่าสูงสุด
__global__ void computeMaxRun(const int* run_lengths, int* max_runs_per_block, int n_edges) {
    __shared__ int sdata[BLOCK_SIZE];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[threadIdx.x] = (i < n_edges) ? run_lengths[i] : 0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] = max(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }

    if (threadIdx.x == 0)
        max_runs_per_block[blockIdx.x] = sdata[0];
}

int findBestEndIdx(const vector<int>& run_lengths, int longestRun) {
    for (int i = run_lengths.size() - 1; i >= 0; --i) {
        if (run_lengths[i] == longestRun && longestRun > 0) {
            return i;
        }
    }
    return 0;
}

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
        int N_edges = N - 1;
        if (N < 2) continue;

        cout << "\n=== Index: " << indexName << " (" << N << " records) ===\n";

        // Allocate device memory
        float* d_prices;
        int* d_trend;
        int* d_segments;
        int* d_run_lengths;
        int* d_max_runs_per_block;
        
        cudaMalloc(&d_prices, N * sizeof(float));
        cudaMalloc(&d_trend, N_edges * sizeof(int));
        cudaMalloc(&d_segments, N_edges * sizeof(int));
        cudaMalloc(&d_run_lengths, N_edges * sizeof(int));
        
        int threads = BLOCK_SIZE;
        int blocks = (N_edges + threads - 1) / threads;
        cudaMalloc(&d_max_runs_per_block, blocks * sizeof(int));


        cudaMemcpy(d_prices, prices.data(), N * sizeof(float), cudaMemcpyHostToDevice);

        // Timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        computeTrend<<<blocks, threads>>>(d_prices, d_trend, N);
        computeRunLengths<<<blocks, threads>>>(d_trend, d_run_lengths, N_edges);
        computeMaxRun<<<blocks, threads>>>(d_run_lengths, d_max_runs_per_block, N_edges);


        // Copy results back
        vector<int> max_runs_per_block_h(blocks);
        cudaMemcpy(max_runs_per_block_h.data(), d_max_runs_per_block, 
                   blocks * sizeof(int), cudaMemcpyDeviceToHost);

        int longestRun = 0;
        for (int max_run : max_runs_per_block_h) {
            longestRun = max(longestRun, max_run);
        }
        
        vector<int> run_lengths_h(N_edges);
        cudaMemcpy(run_lengths_h.data(), d_run_lengths, 
                   N_edges * sizeof(int), cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        int bestEndIdx = findBestEndIdx(run_lengths_h, longestRun);
        int dateStartIdx = (longestRun > 0) ? bestEndIdx - longestRun + 1 : 0;
        int dateEndIdx = (longestRun > 0) ? bestEndIdx + 1 : 0;
        
        float deviceMs = 0;
        cudaEventElapsedTime(&deviceMs, start, stop);
        // Output results
        cout << "Device time: " << deviceMs << " ms\n";
        cout << "Longest consecutive uptrend edges: " << longestRun << "\n";
        cout << "Longest consecutive uptrend days: " << (longestRun + 1) << "\n";

        if (!dates.empty() && longestRun > 0) {
            cout << "Date Range: " << dates[dateStartIdx]
                 << " -> " << dates[dateEndIdx] << "\n";
        } else if (longestRun == 0) {
            cout << "Date Range: No consecutive uptrend found.\n";
        }

        // Cleanup
        cudaFree(d_prices);
        cudaFree(d_trend);
        cudaFree(d_segments);
        cudaFree(d_run_lengths);
        cudaFree(d_max_runs_per_block);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    return 0;
}