#!/usr/bin/env python3
"""
DeepX OCR Server 测试脚本
"""

import requests
import base64
import json
import sys
import argparse
import time
import threading
from pathlib import Path


def test_health(base_url):
    """测试健康检查接口"""
    print("\n=== Testing /health endpoint ===")
    url = f"{base_url}/health"
    
    try:
        response = requests.get(url, timeout=5)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_ocr_base64(base_url, image_path, token="test_token_12345", visualize=False):
    """测试 OCR 接口（Base64 方式）"""
    print(f"\n=== Testing /ocr endpoint with Base64 ===")
    print(f"Image: {image_path}")
    
    # 读取图片并转换为 Base64
    try:
        with open(image_path, "rb") as f:
            image_data = f.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
        print(f"Image size: {len(image_data)} bytes")
        print(f"Base64 length: {len(image_base64)} characters")
    except Exception as e:
        print(f"Error reading image: {e}")
        return False
    
    # 构建请求
    url = f"{base_url}/ocr"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"token {token}"
    }
    data = {
        "file": image_base64,
        "fileType": 1,
        "useDocOrientationClassify": False,
        "useDocUnwarping": False,
        "useTextlineOrientation": False,
        "textDetLimitSideLen": 960,
        "textDetThresh": 0.3,
        "textDetBoxThresh": 0.6,
        "textDetUnclipRatio": 1.5,
        "textRecScoreThresh": 0.0,
        "visualize": visualize
    }
    
    # 发送请求
    try:
        print(f"\nSending request to {url}...")
        response = requests.post(url, headers=headers, json=data, timeout=30)
        print(f"Status Code: {response.status_code}")
        
        result = response.json()
        print(f"\n=== Response ===")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # 提取并显示识别结果
        if result.get("errorCode") == 0:
            print(f"\n=== OCR Results ===")
            ocr_results = result.get("result", {}).get("ocrResults", [])
            print(f"Total text boxes detected: {len(ocr_results)}")
            
            # 不打印每个文本内容
            # for i, item in enumerate(ocr_results, 1):
            #     text = item.get("prunedResult", "")
            #     score = item.get("score", 0.0)
            #     print(f"{i}. Text: {text}")
            #     print(f"   Confidence: {score:.3f}")
            #     
            #     if "ocrImage" in item:
            #         print(f"   Visualization: {base_url}{item['ocrImage']}")
            
            return True
        else:
            print(f"\nError: {result.get('errorMsg')}")
            return False
            
    except requests.exceptions.Timeout:
        print("Error: Request timeout (30s)")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_ocr_url(base_url, image_url, token="test_token_12345", visualize=False):
    """测试 OCR 接口（URL 方式）"""
    print(f"\n=== Testing /ocr endpoint with URL ===")
    print(f"Image URL: {image_url}")
    
    # 构建请求
    url = f"{base_url}/ocr"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"token {token}"
    }
    data = {
        "file": image_url,
        "visualize": visualize
    }
    
    # 发送请求
    try:
        print(f"\nSending request to {url}...")
        response = requests.post(url, headers=headers, json=data, timeout=30)
        print(f"Status Code: {response.status_code}")
        
        result = response.json()
        print(f"\n=== Response ===")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # 提取并显示识别结果
        if result.get("errorCode") == 0:
            print(f"\n=== OCR Results ===")
            ocr_results = result.get("result", {}).get("ocrResults", [])
            print(f"Total text boxes detected: {len(ocr_results)}")
            
            # 不打印每个文本内容
            # for i, item in enumerate(ocr_results, 1):
            #     text = item.get("prunedResult", "")
            #     score = item.get("score", 0.0)
            #     print(f"{i}. Text: {text}")
            #     print(f"   Confidence: {score:.3f}")
            
            return True
        else:
            print(f"\nError: {result.get('errorMsg')}")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_performance(base_url, image_path, num_requests=100, token="test_token_12345", visualize=False):
    """性能测试：发送多次请求并计算FPS"""
    print(f"\n=== Performance Test ===")
    print(f"Image: {image_path}")
    print(f"Number of requests: {num_requests}")
    print(f"Visualization: {visualize}")
    
    # 读取图片并转换为 Base64
    try:
        with open(image_path, "rb") as f:
            image_data = f.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
        print(f"Image size: {len(image_data)} bytes")
    except Exception as e:
        print(f"Error reading image: {e}")
        return False
    
    # 构建请求
    url = f"{base_url}/ocr"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"token {token}"
    }
    data = {
        "file": image_base64,
        "fileType": 1,
        "useDocOrientationClassify": False,
        "useDocUnwarping": False,
        "useTextlineOrientation": False,
        "textDetLimitSideLen": 960,
        "textDetThresh": 0.3,
        "textDetBoxThresh": 0.6,
        "textDetUnclipRatio": 1.5,
        "textRecScoreThresh": 0.0,
        "visualize": visualize
    }
    
    print(f"\nSending {num_requests} requests to {url}...")
    print("Progress: ", end="", flush=True)
    
    success_count = 0
    error_count = 0
    total_boxes = 0
    
    start_time = time.time()
    
    for i in range(num_requests):
        try:
            # 注释掉超时逻辑，等待服务器处理完成
            response = requests.post(url, headers=headers, json=data)  # 移除 timeout
            
            if response.status_code == 200:
                result = response.json()
                if result.get("errorCode") == 0:
                    success_count += 1
                    ocr_results = result.get("result", {}).get("ocrResults", [])
                    total_boxes += len(ocr_results)
                else:
                    error_count += 1
            else:
                error_count += 1
                
        except Exception as e:
            error_count += 1
            print(f"\nError on request {i+1}: {e}")
        
        # 显示进度
        if (i + 1) % 10 == 0:
            print(f"{i+1}...", end="", flush=True)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\n\n=== Performance Results ===")
    print(f"Total requests: {num_requests}")
    print(f"Successful: {success_count}")
    print(f"Failed: {error_count}")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Average time per request: {elapsed_time/num_requests:.3f} seconds")
    print(f"Throughput (FPS): {success_count/elapsed_time:.2f} requests/second")
    print(f"Total text boxes detected: {total_boxes}")
    print(f"Average boxes per image: {total_boxes/success_count if success_count > 0 else 0:.1f}")
    
    return success_count == num_requests


def test_auth_failure(base_url):
    """测试认证失败情况"""
    print("\n=== Testing authentication failure ===")
    url = f"{base_url}/ocr"
    headers = {
        "Content-Type": "application/json"
        # 故意不添加 Authorization header
    }
    data = {
        "file": "dummy_base64"
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=5)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        # 应该返回 401
        return response.status_code == 401
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="DeepX OCR Server Test Client")
    parser.add_argument("--host", default="localhost", help="Server host (default: localhost)")
    parser.add_argument("--port", type=int, default=8080, help="Server port (default: 8080)")
    parser.add_argument("--image", help="Path to test image file")
    parser.add_argument("--url", help="URL of test image")
    parser.add_argument("--token", default="test_token_12345", help="Access token")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization")
    parser.add_argument("--test-all", action="store_true", help="Run all tests")
    parser.add_argument("--perf", action="store_true", help="Run performance test")
    parser.add_argument("--num-requests", type=int, default=100, help="Number of requests for performance test (default: 100)")
    
    args = parser.parse_args()
    base_url = f"http://{args.host}:{args.port}"
    
    print("=" * 60)
    print("DeepX OCR Server Test Client")
    print("=" * 60)
    print(f"Server: {base_url}")
    print(f"Token: {args.token}")
    print("=" * 60)
    
    results = []
    
    # 性能测试模式
    if args.perf:
        if not args.image:
            print("\nError: --image is required for performance test")
            sys.exit(1)
        if not Path(args.image).exists():
            print(f"\nError: Image file not found: {args.image}")
            sys.exit(1)
        
        test_performance(base_url, args.image, args.num_requests, args.token, args.visualize)
        sys.exit(0)
    
    # 测试健康检查
    results.append(("Health Check", test_health(base_url)))
    
    if args.test_all or not args.image and not args.url:
        # 测试认证失败
        results.append(("Auth Failure", test_auth_failure(base_url)))
    
    # 测试 OCR（Base64）
    if args.image:
        if not Path(args.image).exists():
            print(f"\nError: Image file not found: {args.image}")
            sys.exit(1)
        results.append(("OCR (Base64)", test_ocr_base64(
            base_url, args.image, args.token, args.visualize)))
    
    # 测试 OCR（URL）
    if args.url:
        results.append(("OCR (URL)", test_ocr_url(
            base_url, args.url, args.token, args.visualize)))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:20s} {status}")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    sys.exit(0 if passed_count == total_count else 1)


if __name__ == "__main__":
    main()
