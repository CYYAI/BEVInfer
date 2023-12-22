#include "common/inc/flag.h"

namespace bevinfer {
namespace optparse {
ParseFlag::ParseFlag(optparse::Values &input_option) {
  FLAGS_name = input_option["modelName"];
  FLAGS_inference_type = input_option["engineName"];
  FLAGS_model_value = input_option["modelPath"];

  // device & modelType 的设置
  if (FLAGS_inference_type == "kInferenceTypeTensorRt") {
    FLAGS_device_type = "kDeviceTypeCodeCuda:0";
    if (common::end_with(FLAGS_model_value, ".onnx"))
      FLAGS_model_type = "kModelTypeOnnx";
    else
      FLAGS_model_type = "kModelTypeTensorRt";
  } else if (FLAGS_inference_type == "kInferenceTypeOnnxRuntime") {
    FLAGS_device_type =
        "kDeviceTypeCodeX86:0";  // 如果是arm平台，需改为 kDeviceTypeCodeArm:0
    FLAGS_model_type = "kModelTypeOnnx";
  } else {
    ALOG_ERROR("[%s] backend is not support!", FLAGS_inference_type.c_str());
  }

  FLAGS_input_path = input_option["inputPath"];
}

ParseFlag::ParseFlag(const std::string &name, const std::string &engineName,
                     const std::string &model_path) {
  FLAGS_name = name;
  FLAGS_inference_type = engineName;
  FLAGS_model_value = model_path;

  // device & modelType 的设置
  if (FLAGS_inference_type == "kInferenceTypeTensorRt") {
    FLAGS_device_type = "kDeviceTypeCodeCuda:0";
    if (common::end_with(FLAGS_model_value, ".onnx"))
      FLAGS_model_type = "kModelTypeOnnx";
    else
      FLAGS_model_type = "kModelTypeTensorRt";
  } else if (FLAGS_inference_type == "kInferenceTypeOnnxRuntime") {
    FLAGS_device_type =
        "kDeviceTypeCodeX86:0";  // 如果是arm平台，需改为 kDeviceTypeCodeArm:0
    FLAGS_model_type = "kModelTypeOnnx";
  } else {
    ALOG_ERROR("[%s] backend is not support!", FLAGS_inference_type.c_str());
  }
}
//   void showUsage(){}

std::string ParseFlag::getName() { return FLAGS_name; }

base::InferenceType ParseFlag::getInferenceType() {
  return base::stringToInferenceType(FLAGS_inference_type);
}

base::DeviceType ParseFlag::getDeviceType() {
  return base::stringToDeviceType(FLAGS_device_type);
}

base::ModelType ParseFlag::getModelType() {
  return base::stringToModelType(FLAGS_model_type);
}

bool ParseFlag::isPath() { return FLAGS_is_path; }

std::vector<std::string> ParseFlag::getModelValue() {
  std::vector<std::string> model_value;
  std::string model_value_str = FLAGS_model_value;
  if (!base::exists(model_value_str)) {
    ALOG_ERROR("model path:{%s} is not exist", model_value_str.c_str());
    // return model_value;
  }
  std::string::size_type pos1, pos2;
  pos2 = model_value_str.find(",");
  pos1 = 0;
  while (std::string::npos != pos2) {
    model_value.emplace_back(model_value_str.substr(pos1, pos2 - pos1));
    pos1 = pos2 + 1;
    pos2 = model_value_str.find(",", pos1);
  }
  model_value.emplace_back(model_value_str.substr(pos1));
  return model_value;
}

InputType ParseFlag::getInputType() {
  if (FLAGS_input_type == "kInputTypeImage") {
    return kInputTypeImage;
  } else if (FLAGS_input_type == "kInputTypeVideo") {
    return kInputTypeVideo;
  } else if (FLAGS_input_type == "kInputTypeCamera") {
    return kInputTypeCamera;
  } else if (FLAGS_input_type == "kDeviceTypeOther") {
    return kDeviceTypeOther;
  } else {
    return kInputTypeImage;
  }
}
std::string ParseFlag::getInputPath() { return FLAGS_input_path; }
// std::string ParseFlag::getOutputPath() { return FLAGS_output_path; }

int ParseFlag::getNumThread() { return FLAGS_num_thread; }
int ParseFlag::getGpuTuneKernel() { return FLAGS_gpu_tune_kernel; }
base::ShareMemoryType ParseFlag::getShareMemoryType() {
  return base::stringToShareMemoryType(FLAGS_share_memory_mode);
}
base::PrecisionType ParseFlag::getPrecisionType() {
  return base::stringToPrecisionType(FLAGS_precision_type);
}
base::PowerType ParseFlag::getPowerType() {
  return base::stringToPowerType(FLAGS_power_type);
}
std::vector<std::string> ParseFlag::getCachePath() {
  std::vector<std::string> cache_path;
  std::string cache_path_str = FLAGS_cache_path;
  std::string::size_type pos1, pos2;
  pos2 = cache_path_str.find(",");
  pos1 = 0;
  while (std::string::npos != pos2) {
    cache_path.emplace_back(cache_path_str.substr(pos1, pos2 - pos1));
    pos1 = pos2 + 1;
    pos2 = cache_path_str.find(",", pos1);
  }
  cache_path.emplace_back(cache_path_str.substr(pos1));
  return cache_path;
}
std::vector<std::string> ParseFlag::getLibraryPath() {
  std::vector<std::string> library_path;
  std::string library_path_str = FLAGS_library_path;
  std::string::size_type pos1, pos2;
  pos2 = library_path_str.find(",");
  pos1 = 0;
  while (std::string::npos != pos2) {
    library_path.emplace_back(library_path_str.substr(pos1, pos2 - pos1));
    pos1 = pos2 + 1;
    pos2 = library_path_str.find(",", pos1);
  }
  library_path.emplace_back(library_path_str.substr(pos1));
  return library_path;
}

std::vector<std::string> ParseFlag::getAllFileFromDir(std::string dir_path) {
  std::vector<std::string> allFile = {};
  if (base::isDirectory(dir_path)) {
    base::glob(dir_path, "", allFile);
  }
  return allFile;
}
}  // namespace optparse
}  // namespace bevinfer