#include "application/yolov8/yolov8_pipeline.h"

using namespace bevinfer;

int main(int argc, const char **argv) {
  optparse::OptionParser parser;
  // 这个就是你在实现推理时，进行注册的的模型名称
  parser.add_option("-n", "--name")
      .dest("modelName")
      .set_default("BEVINFER_YOLOV8")
      .help("model register name");

  // kInferenceTypeTensorRt & kInferenceTypeOnnxRuntime
  parser.add_option("-e", "--engine")
      .dest("engineName")
      .set_default("kInferenceTypeTensorRt")
      .help("infer engine");

  parser.add_option("-l", "--loopNum")
      .dest("loopNum")
      .set_default("1")
      .help("infer engine");

  parser.add_option("-m", "--model_path")
      .dest("modelPath")
      .set_default("/home/yyds/Codes/BEVInfer/workspace/models/yolov8s.engine")
      .help("input model path");

  parser.add_option("-i", "--input_path")
      .dest("inputPath")
      .set_default("/home/yyds/Codes/BEVInfer/workspace/test_data/sample.jpg")
      .help("input img path");

  parser.add_option("-o", "--output_path")
      .dest("outputPath")
      .set_default(
          "/home/yyds/Codes/BEVInfer/workspace/test_data/sample_res.jpg")
      .help("output img path");

  optparse::Values options = parser.parse_args(argc, argv);
  bevinfer::application::YOLOV8APP yolov8_app(options);
  yolov8_app.run();

  return 0;
}
