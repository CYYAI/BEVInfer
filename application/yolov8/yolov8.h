
#ifndef _BEVINFER_APPLICATION_DETECT_YOLOV8_H_
#define _BEVINFER_APPLICATION_DETECT_YOLOV8_H_

#include "common/inc/common_include.h"
#include "common/inc/graph_include.h"  // 导入推理所需的包
#include "nndeploy/model/detect/result.h"
#include "nndeploy/model/detect/util.h"
#include "nndeploy/model/preprocess/opencv_convert.h"
#include "nndeploy/model/preprocess/util.h"

namespace bevinfer {
namespace application {
using namespace nndeploy;
#define BEVINFER_YOLOV8 "BEVINFER_YOLOV8"

class NNDEPLOY_CC_API YoloPreParam : public base::Param {
 public:
  base::PixelType src_pixel_type_ = base::kPixelTypeBGR;
  base::PixelType dst_pixel_type_ = base::kPixelTypeRGB;
  base::InterpType interp_type_ = base::kInterpTypeLinear;
  int h_ = 640;
  int w_ = 640;
  float scale_[4] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f,
                     1.0f / 255.0f};
  float mean_[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float std_[4] = {1.0f, 1.0f, 1.0f, 1.0f};
};

class NNDEPLOY_CC_API YoloPreProcess : public dag::Node {
 public:
  YoloPreProcess(const std::string &name, dag::Edge *input, dag::Edge *output)
      : Node(name, input, output) {
    param_ = std::make_shared<YoloPreParam>();
  }
  virtual ~YoloPreProcess() {}

  virtual base::Status run();
};

class NNDEPLOY_CC_API YoloPostParam : public base::Param {
 public:
  float score_threshold_ = 0.5;
  float nms_threshold_ = 0.45;
  int num_classes_ = 80;

  int model_h_ = 640;
  int model_w_ = 640;
};

class NNDEPLOY_CC_API YoloPostProcess : public dag::Node {
 public:
  YoloPostProcess(const std::string &name, dag::Edge *input, dag::Edge *output)
      : Node(name, input, output) {
    param_ = std::make_shared<YoloPostParam>();
  }
  virtual ~YoloPostProcess() {}

  virtual base::Status run();
};

extern NNDEPLOY_CC_API dag::Graph *createYoloV8Graph(
    const std::string &name, base::InferenceType inference_type,
    base::DeviceType device_type, dag::Edge *input, dag::Edge *output,
    base::ModelType model_type, bool is_path,
    std::vector<std::string> model_value);

}  // namespace application
}  // namespace bevinfer

#endif /* _BEVINFER_APPLICATION_DETECT_YOLOV8_H_ */
