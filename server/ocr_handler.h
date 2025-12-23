#pragma once

#include "pipeline/ocr_pipeline.h"
#include "file_handler.h"
#include "json_response.h"
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <memory>
#include <string>

using json = nlohmann::json;

namespace ocr_server {

/**
 * @brief OCR请求参数结构
 */
struct OCRRequest {
    std::string file;                       // Base64编码或URL
    int fileType = 1;                       // 1: 图像, 0: PDF(未实现)
    bool useDocOrientationClassify = false; // 文档方向矫正
    bool useDocUnwarping = false;           // 图片扭曲矫正
    bool useTextlineOrientation = false;    // 文本行方向矫正
    int textDetLimitSideLen = 64;           // 图像长边限制
    double textDetThresh = 0.3;             // 检测像素阈值
    double textDetBoxThresh = 0.6;          // 检测框阈值
    double textDetUnclipRatio = 1.5;        // 检测扩张系数
    double textRecScoreThresh = 0.0;        // 识别置信度阈值
    bool visualize = false;                 // 是否开启可视化
    
    /**
     * @brief 从JSON解析请求参数
     */
    static OCRRequest FromJson(const json& j);
    
    /**
     * @brief 验证请求参数
     */
    bool Validate(std::string& error_msg) const;
};

/**
 * @brief OCR请求处理器
 */
class OCRHandler {
public:
    /**
     * @brief 构造函数
     * @param pipeline_config OCR Pipeline配置
     * @param vis_output_dir 可视化图片输出目录
     * @param vis_url_prefix 可视化图片URL前缀
     */
    OCRHandler(
        const ocr::OCRPipelineConfig& pipeline_config,
        const std::string& vis_output_dir = "output/vis",
        const std::string& vis_url_prefix = "/static/vis"
    );
    
    /**
     * @brief 处理OCR请求
     * @param request OCR请求参数
     * @param response_json 输出的JSON响应
     * @return HTTP状态码
     */
    int HandleRequest(const OCRRequest& request, json& response_json);
    
private:
    /**
     * @brief 从请求参数创建OCR Pipeline配置
     */
    ocr::OCRPipelineConfig CreatePipelineConfig(const OCRRequest& request) const;
    
    /**
     * @brief 加载输入图像（Base64或URL）
     */
    bool LoadInputImage(const OCRRequest& request, cv::Mat& image, std::string& error_msg);
    
    std::shared_ptr<ocr::OCRPipeline> base_pipeline_;  // 基础Pipeline实例
    ocr::OCRPipelineConfig base_config_;               // 基础配置
    std::string vis_output_dir_;                       // 可视化输出目录
    std::string vis_url_prefix_;                       // 可视化URL前缀
};

} // namespace ocr_server
