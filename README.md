

## Parallel Stock Trend Analysis using CUDA (Map Pattern)

## 📂 Folder Structure

```
project_root/
│
├── data/
│   └── indexProcessed.csv       # Input dataset
│
├── src/
│   └── trend_parallel.cu        # Parallel source code
|   └── trend_seq.cpp            # Sequential source code
│
├── presentation/
│   └── cudaproject.pdf          # Presentation file
│
├── README.md                    # This documentation file

```

---

## How to Compile and Run

### 1 Compile

```bash
nvcc src/trend_parallel.cu -o trend_parallel
```

### 2️ Run

```bash
./trend_parallel
```

### 3️ Expected Output (Example)

```
CSV loaded in 150 ms
Found 5 index groups.

=== Index: SET50 (250 records) ===
Device time (trend calc): 0.051 ms
Longest consecutive uptrend edges: 7
Longest consecutive uptrend days: 8
Date Range: 2023-02-01 -> 2023-02-08
```

---

## 🧪 Dataset Information

* **File:** `indexProcessed.csv`
* **Columns used:**

  * Column 0 → `IndexName`
  * Column 1 → `Date`
  * Column 8 → `Close Price`

---

## 📈 Performance Evaluation

| Operation         | Description                                | Time (ms) |
| ----------------- | ------------------------------------------ | --------- |
| CSV Load          | Sequential file I/O                        | 120       |
| Trend Computation | CUDA kernel execution                      | 0.05      |
| **Speed-up**      | Compared to CPU sequential trend detection | ~30×      |

---

