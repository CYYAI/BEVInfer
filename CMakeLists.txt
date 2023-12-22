cmake_minimum_required(VERSION 3.12)

project(nndeploy)

# set(CMAKE_BUILD_TYPE "Release") # 运行项目时开启
set(CMAKE_BUILD_TYPE "Debug") # debug时开启

# 这个是在主目录的cmake中起作用的
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 变量具体作用，请移步nndeploy
set(ENABLE_NNDEPLOY_OPENCV ON)
set(ENABLE_NNDEPLOY_DEVICE_X86 ON) # 开启x86环境
set(ENABLE_NNDEPLOY_DEVICE_ARM OFF) # 如果要用arm再开启
set(ENABLE_NNDEPLOY_DEVICE_CUDA ON) # 开启cuda
set(ENABLE_NNDEPLOY_DEVICE_CUDNN ON) # 开启cudnn
set(ENABLE_NNDEPLOY_MODEL_DETECT ON)

# 这里我只用tensorrt和onnxruntime两种框架，所以我只开启两种，如果你想用多种框架，建议看nndeploy使用方式
set(ENABLE_NNDEPLOY_INFERENCE_TENSORRT "/home/dell/Codes/week_06/aideploy/dependencies/thirdparty/x86/TensorRT-8.4.3.1")
set(ENABLE_NNDEPLOY_INFERENCE_ONNXRUNTIME "/home/dell/Codes/week_06/aideploy/dependencies/thirdparty/x86/onnxruntime-linux-x64-gpu-1.15.0")

#define
if(${ENABLE_NNDEPLOY_OPENCV} MATCHES "OFF")
else()
  add_definitions(-DENABLE_NNDEPLOY_OPENCV)
endif()

add_subdirectory(thirdparty/nndeploy)
include_directories(common)
include_directories(thirdparty/nndeploy/include)

file(GLOB_RECURSE bev_common_cpps
    ${PROJECT_SOURCE_DIR}/common/*.cpp
)
# cuda_add_library(utils_cu_cpp SHARED ${bev_common_cpps})
add_library(bev_common SHARED ${bev_common_cpps})

file(GLOB_RECURSE application_cpps
    ${PROJECT_SOURCE_DIR}/application/*.cpp
)

add_executable(bev_demo bev_main.cpp ${application_cpps})
target_link_libraries(bev_demo
  bev_common
  nndeploy    
  pthread)