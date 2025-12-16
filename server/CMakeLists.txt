# Server CMakeLists.txt
cmake_minimum_required(VERSION 3.15)

# 查找必需的依赖
find_package(CURL REQUIRED)

# 查找UUID库（使用pkg-config）
find_package(PkgConfig REQUIRED)
pkg_check_modules(UUID REQUIRED uuid)

# Server源文件
set(SERVER_SOURCES
    server_main.cpp
    ocr_handler.cpp
    json_response.cpp
    file_handler.cpp
    ${CMAKE_SOURCE_DIR}/3rd-party/cpp-base64/base64.cpp  # cpp-base64实现
)

# 创建server可执行文件
add_executable(ocr_server ${SERVER_SOURCES})

# 包含目录
target_include_directories(ocr_server PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/3rd-party/crow/include
    ${CMAKE_SOURCE_DIR}/3rd-party/cpp-base64
    ${CMAKE_SOURCE_DIR}/3rd-party/json/include
    ${CMAKE_SOURCE_DIR}/3rd-party/spdlog/include
    ${OpenCV_INCLUDE_DIRS}
    ${UUID_INCLUDE_DIRS}
)

# 链接库
target_link_libraries(ocr_server PRIVATE
    ocr_pipeline           # 我们的OCR Pipeline库
    ocr_detection
    ocr_classification
    ocr_recognition
    ocr_preprocessing
    ocr_common
    ${OpenCV_LIBS}
    CURL::libcurl
    ${UUID_LIBRARIES}
    pthread
)

# C++17标准
set_target_properties(ocr_server PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# 打印信息
message(STATUS "Server executable: ocr_server")
message(STATUS "CURL libraries: ${CURL_LIBRARIES}")
message(STATUS "UUID libraries: ${UUID_LIBRARIES}")
