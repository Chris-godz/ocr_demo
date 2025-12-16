# DXNN-OCR Benchmark Report

**Test Configuration**:
- Model: PP-OCR v5 (DEEPX NPU acceleration)
- Total Images Tested: 20
- Success Rate: 100.0%

**Test Results**:
| Filename | Inference Time (ms) | FPS | CPS (chars/s) | Accuracy (%) |
|---|---|---|---|---|
| `image_10.png` | 172.70 | 5.79 | **7267.05** | **99.50** |
| `image_11.png` | 172.70 | 5.79 | **15616.92** | **93.83** |
| `image_12.png` | 172.70 | 5.79 | **12831.70** | **71.53** |
| `image_13.png` | 172.70 | 5.79 | **1163.89** | **98.48** |
| `image_14.png` | 172.70 | 5.79 | **12217.91** | **86.85** |
| `image_15.png` | 172.70 | 5.79 | **23503.55** | **79.78** |
| `image_16.png` | 172.70 | 5.79 | **885.94** | **95.83** |
| `image_17.png` | 172.70 | 5.79 | **1418.67** | **95.18** |
| `image_18.png` | 172.70 | 5.79 | **10527.09** | **99.33** |
| `image_19.png` | 172.70 | 5.79 | **12206.33** | **95.68** |
| `image_1.png` | 172.70 | 5.79 | **121.60** | **42.86** |
| `image_20.png` | 172.70 | 5.79 | **9172.12** | **95.29** |
| `image_2.png` | 172.70 | 5.79 | **903.31** | **38.00** |
| `image_3.png` | 172.70 | 5.79 | **214.25** | **50.00** |
| `image_4.png` | 172.70 | 5.79 | **787.50** | **44.64** |
| `image_5.png` | 172.70 | 5.79 | **173.71** | **95.24** |
| `image_6.png` | 172.70 | 5.79 | **22432.31** | **96.30** |
| `image_7.png` | 172.70 | 5.79 | **6410.06** | **83.33** |
| `image_8.png` | 172.70 | 5.79 | **7070.17** | **93.13** |
| `image_9.png` | 172.70 | 5.79 | **13555.51** | **95.13** |
| **Average** | **172.70** | **5.79** | **7923.98** | **82.50** |

**Performance Summary**:
- Average Inference Time: **172.70 ms**
- Average FPS: **5.79**
- Average CPS: **7923.98 chars/s**
- Total Characters Detected: **27369**
- Total Processing Time: **3453.95 ms**
- Average Character Accuracy: **82.50%**
- Success Rate: **100.0%** (20/20 images)
