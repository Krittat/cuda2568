// trend_parallel_cuda.cu
// Pure CUDA version (no Thrust) implementing Map, Scan (prefix), Reduce
// Build: nvcc -O2 -std=c++14 trend_parallel_cuda.cu -o trend_parallel_cuda

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>

using namespace std;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "CUDA Error: " << cudaGetErrorString(err) \
                 << " at " << __FILE__ << ":" << __LINE__ << endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// อ่านราคาปิดจากไฟล์ CSV (column 5)
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
        int colIndex = 0;
        float closePrice = 0;
        bool got = false;
        while (getline(ss, token, ',')) {
            if (colIndex == 5) {
                try { closePrice = stof(token); got = true; } catch(...) { got = false; }
                break;
            }
            ++colIndex;
        }
        if (got) prices.push_back(closePrice);
    }
    return prices;
}

// -----------------------
// Kernel: map -> trend (1 if next day > today else 0)
// trend length = n-1
// -----------------------
__global__ void computeTrend(const float* prices, int* trend, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n - 1) {
        trend[i] = (prices[i + 1] > prices[i]) ? 1 : 0;
    }
}

// -------------------------------------------------------------------
// Kernel: block-wise inclusive scan using Hillis-Steele in shared mem
// - processes one block's contiguous chunk
// - writes per-element inclusive prefix to out (same length as in)
// - writes block sum (last element) to blockSums[blockIdx]
// NOTE: blockDim.x should be chosen (e.g., 256)
// -------------------------------------------------------------------
__global__ void blockInclusiveScan(const int* in, int* out, int* blockSums, int n) {
    extern __shared__ int sdata[]; // size: blockDim.x
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // load or zero
    int val = (gid < n) ? in[gid] : 0;
    sdata[tid] = val;
    __syncthreads();

    // Hillis-Steele inclusive scan (simple, robust for block sizes up to 1024)
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        int t = 0;
        if (tid >= offset) t = sdata[tid - offset];
        __syncthreads();
        sdata[tid] += t;
        __syncthreads();
    }

    // write out (only if within n)
    if (gid < n) out[gid] = sdata[tid];

    // last element of the block is block sum
    if (tid == blockDim.x - 1) {
        // But careful: if block is partial at end, find actual last index
        int lastIndex = blockIdx.x * blockDim.x + (blockDim.x - 1);
        if (lastIndex < n) blockSums[blockIdx.x] = sdata[tid];
        else {
            // find actual last valid index inside block
            int validLast = n - 1 - blockIdx.x * blockDim.x;
            if (validLast <= 0) blockSums[blockIdx.x] = 0;
            else blockSums[blockIdx.x] = sdata[validLast - 1];
        }
    }
}

// -------------------------------------------------------------------
// Kernel: add block offsets to out array
// offset array length = numBlocks
// For block b, add offsets[b] to each element in block b
// -------------------------------------------------------------------
__global__ void addBlockOffsets(int* out, const int* offsets, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        int blockId = blockIdx.x;
        out[gid] += offsets[blockId];
    }
}

// -------------------------------------------------------------------
// Kernel: find longest consecutive ones (per-block)
// Strategy: each thread scans its stride to compute local max run, then intra-block reduce
// Writes block maximum to blockMax[blockIdx]
// -------------------------------------------------------------------
__global__ void findLongestRun(const int* trend, int* blockMax, int n) {
    extern __shared__ int smax[]; // size blockDim.x
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    // Each thread scans elements in a strided manner computing local max run length
    int run = 0;
    int localMax = 0;
    for (int i = gid; i < n; i += stride) {
        if (trend[i] == 1) {
            run++;
            if (run > localMax) localMax = run;
        } else {
            run = 0;
        }
    }
    smax[tid] = localMax;
    __syncthreads();

    // intra-block reduction (max)
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            if (smax[tid + offset] > smax[tid]) smax[tid] = smax[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) blockMax[blockIdx.x] = smax[0];
}

// -----------------------
// host helper: exclusive prefix on vector<int>
// -----------------------
void exclusivePrefixHost(vector<int>& v) {
    int s = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        int tmp = v[i];
        v[i] = s;
        s += tmp;
    }
}

// -----------------------
// main
// -----------------------
int main() {
    string filename = "data/indexData.csv";


    auto host_start_total = chrono::high_resolution_clock::now();

    vector<float> prices = readPricesFromCSV(filename);
    int N = (int)prices.size();
    if (N < 2) {
        cerr << "Not enough data\n";
        return 1;
    }
    cout << "Loaded " << N << " prices.\n";

    int M = N - 1; // length of trend array

    // allocate device
    float* d_prices = nullptr;
    int* d_trend = nullptr;
    CUDA_CHECK(cudaMalloc(&d_prices, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_trend, M * sizeof(int)));

    // copy prices
    CUDA_CHECK(cudaMemcpy(d_prices, prices.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // device timing
    cudaEvent_t devStart, devEnd;
    CUDA_CHECK(cudaEventCreate(&devStart));
    CUDA_CHECK(cudaEventCreate(&devEnd));
    CUDA_CHECK(cudaEventRecord(devStart));

    // --- Map: computeTrend ---
    int threads = 256;
    int blocks = (M + threads - 1) / threads;
    computeTrend<<<blocks, threads>>>(d_prices, d_trend, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Scan: block-wise inclusive scan ---
    // allocate blockSums (numBlocks = blocks)
    int numBlocks = blocks;
    int* d_blockSums = nullptr;
    CUDA_CHECK(cudaMalloc(&d_blockSums, numBlocks * sizeof(int)));
    // output prefix same length as trend
    int* d_prefix = nullptr;
    CUDA_CHECK(cudaMalloc(&d_prefix, M * sizeof(int)));

    // per-block scan
    size_t shmem = threads * sizeof(int);
    blockInclusiveScan<<<numBlocks, threads, shmem>>>(d_trend, d_prefix, d_blockSums, M);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // copy block sums back to host, do exclusive prefix on them, and copy offsets back
    vector<int> h_blockSums(numBlocks);
    CUDA_CHECK(cudaMemcpy(h_blockSums.data(), d_blockSums, numBlocks * sizeof(int), cudaMemcpyDeviceToHost));
    vector<int> h_offsets = h_blockSums;
    exclusivePrefixHost(h_offsets); // now offsets[b] is sum of all previous blocks
    // copy offsets to device
    int* d_offsets = nullptr;
    CUDA_CHECK(cudaMalloc(&d_offsets, numBlocks * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_offsets, h_offsets.data(), numBlocks * sizeof(int), cudaMemcpyHostToDevice));

    // add offsets per block
    addBlockOffsets<<<numBlocks, threads>>>(d_prefix, d_offsets, M);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Reduce: findLongestRun per-block then host combine ---
    int* d_blockMax = nullptr;
    CUDA_CHECK(cudaMalloc(&d_blockMax, numBlocks * sizeof(int)));
    size_t shmem2 = threads * sizeof(int);
    findLongestRun<<<numBlocks, threads, shmem2>>>(d_trend, d_blockMax, M);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // copy blockMax back and find global max
    vector<int> h_blockMax(numBlocks);
    CUDA_CHECK(cudaMemcpy(h_blockMax.data(), d_blockMax, numBlocks * sizeof(int), cudaMemcpyDeviceToHost));
    int longest_edges = 0;
    for (int v : h_blockMax) if (v > longest_edges) longest_edges = v;

    // device time end
    CUDA_CHECK(cudaEventRecord(devEnd));
    CUDA_CHECK(cudaEventSynchronize(devEnd));
    float deviceMs = 0;
    CUDA_CHECK(cudaEventElapsedTime(&deviceMs, devStart, devEnd));

    auto host_end_total = chrono::high_resolution_clock::now();
    auto host_total_ms = chrono::duration_cast<chrono::milliseconds>(host_end_total - host_start_total).count();

    cout << "Device kernels time (approx): " << deviceMs << " ms\n";
    cout << "Host total time (including I/O): " << host_total_ms << " ms\n";

    cout << "Longest consecutive increasing edges: " << longest_edges << "\n";
    cout << "Longest consecutive increasing days: " << (longest_edges > 0 ? longest_edges + 1 : 1) << "\n";

    // show sample prefix (first 20)
    vector<int> h_prefix(min(M, 20));
    CUDA_CHECK(cudaMemcpy(h_prefix.data(), d_prefix, h_prefix.size() * sizeof(int), cudaMemcpyDeviceToHost));
    cout << "Sample prefix (first up to 20):\n";
    for (size_t i = 0; i < h_prefix.size(); ++i) cout << h_prefix[i] << " ";
    cout << "\n";

    // cleanup
    CUDA_CHECK(cudaFree(d_prices));
    CUDA_CHECK(cudaFree(d_trend));
    CUDA_CHECK(cudaFree(d_prefix));
    CUDA_CHECK(cudaFree(d_blockSums));
    CUDA_CHECK(cudaFree(d_offsets));
    CUDA_CHECK(cudaFree(d_blockMax));
    CUDA_CHECK(cudaEventDestroy(devStart));
    CUDA_CHECK(cudaEventDestroy(devEnd));

    return 0;
}
