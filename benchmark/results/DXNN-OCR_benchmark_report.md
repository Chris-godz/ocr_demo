# DXNN-OCR Benchmark Report

**Test Configuration**:
- Model: PP-OCR v5 (DEEPX NPU acceleration)
- Total Images Tested: 20
- Success Rate: 100.0%

**Test Results**:
| Filename | Inference Time (ms) | FPS | CPS (chars/s) | Accuracy (%) |
|---|---|---|---|---|
| `image_10.png` | 1185.15 | 0.84 | **1065.69** | **100.00** |
| `image_11.png` | 2499.02 | 0.40 | **1113.24** | **98.90** |
| `image_12.png` | 2118.95 | 0.47 | **1098.18** | **70.98** |
| `image_13.png` | 982.17 | 1.02 | **132.36** | **65.15** |
| `image_14.png` | 1853.70 | 0.54 | **1040.08** | **89.60** |
| `image_15.png` | 3205.78 | 0.31 | **1426.80** | **99.11** |
| `image_16.png` | 961.64 | 1.04 | **145.58** | **87.50** |
| `image_17.png` | 546.24 | 1.83 | **477.81** | **100.00** |
| `image_18.png` | 1445.31 | 0.69 | **1251.64** | **98.99** |
| `image_19.png` | 1637.67 | 0.61 | **1295.13** | **98.07** |
| `image_1.png` | 921.39 | 1.09 | **26.05** | **57.14** |
| `image_20.png` | 1532.44 | 0.65 | **1019.94** | **96.47** |
| `image_2.png` | 949.81 | 1.05 | **108.44** | **44.00** |
| `image_3.png` | 934.16 | 1.07 | **38.54** | **35.71** |
| `image_4.png` | 946.80 | 1.06 | **126.74** | **42.86** |
| `image_5.png` | 905.04 | 1.10 | **14.36** | **42.86** |
| `image_6.png` | 2493.04 | 0.40 | **1563.96** | **97.35** |
| `image_7.png` | 835.93 | 1.20 | **1324.27** | **90.12** |
| `image_8.png` | 1487.21 | 0.67 | **790.07** | **86.01** |
| `image_9.png` | 1930.40 | 0.52 | **1194.57** | **92.96** |
| **Average** | **1468.59** | **0.68** | **942.43** | **79.69** |

**Performance Summary**:
- Average Inference Time: **1468.59 ms**
- Average FPS: **0.68**
- Average CPS: **942.43 chars/s**
- Total Characters Detected: **27681**
- Total Processing Time: **29371.86 ms**
- Average Character Accuracy: **79.69%**
- Success Rate: **100.0%** (20/20 images)
