
#ifndef _AIDEPLOY_COMMON_FLAG_INCLUDE_H_
#define _AIDEPLOY_COMMON_FLAG_INCLUDE_H_

#include <string>
#include <vector>

#include "argparse.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/file.h"
#include "plog.h"

namespace bevinfer {
namespace optparse {
using namespace nndeploy;

enum InputType : int {
  kInputTypeImage = 0x0000,
  kInputTypeVideo,
  kInputTypeCamera,

  kDeviceTypeOther,
};

class ParseFlag {
 public:
  ParseFlag(optparse::Values &input_option);
  ParseFlag(const std::string &name, const std::string &engineName,
            const std::string &model_path);
  ~ParseFlag() = default;
  //   void showUsage();
  std::string getName();
  base::InferenceType getInferenceType();
  base::DeviceType getDeviceType();
  base::ModelType getModelType();
  bool isPath();
  std::vector<std::string> getModelValue();
  InputType getInputType();
  std::string getInputPath();
  std::string getOutputPath();
  int getNumThread();
  int getGpuTuneKernel();
  base::ShareMemoryType getShareMemoryType();
  base::PrecisionType getPrecisionType();
  base::PowerType getPowerType();
  std::vector<std::string> getCachePath();
  std::vector<std::string> getLibraryPath();
  std::vector<std::string> getAllFileFromDir(std::string dir_path);

 private:
  std::string FLAGS_name;
  std::string FLAGS_inference_type;
  std::string FLAGS_device_type;
  std::string FLAGS_model_type;
  bool FLAGS_is_path = true;
  std::string FLAGS_model_value = "";
  std::string FLAGS_input_type = "KInputTypeImage";  // 目前暂只支持图片的输入
  std::string FLAGS_input_path = "";
  int FLAGS_num_thread = 4;
  int FLAGS_gpu_tune_kernel = -1;
  std::string FLAGS_share_memory_mode = "";
  std::string FLAGS_precision_type = "";
  std::string FLAGS_power_type = "";
  std::string FLAGS_cache_path = "";
  std::string FLAGS_library_path = "";
};
}  // namespace optparse
}  // namespace bevinfer

#endif /* _AIDEPLOY_COMMON_FLAG_INCLUDE_H_ */
