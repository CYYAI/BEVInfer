
#ifndef _BEVINFER_APPLICATION_YOLOV8_PIPELINE_H_
#define _BEVINFER_APPLICATION_YOLOV8_PIPELINE_H_

#include "common/inc/common_include.h"
#include "common/inc/infer_graph.h"
#include "common/inc/infer_graph_util.h"
#include "nndeploy/model/detect/result.h"
#include "utils/inc/show_result.h"
#include "yolov8.h"

namespace bevinfer {
namespace application {
class YOLOV8APP : public infer::ApplicationBaseInfer {
 public:
  YOLOV8APP(optparse::Values input_options) : options(input_options) {}

 private:
  optparse::Values options;

 public:
  base::Status run() {
    optparse::ParseFlag flag(options);
    base::InferenceType inference_type = flag.getInferenceType();
    base::DeviceType device_type = flag.getDeviceType();
    base::ModelType model_type = flag.getModelType();
    bool is_path = flag.isPath();
    std::vector<std::string> model_value = flag.getModelValue();

    // 创建输入和输出的对象，名称随意起只要不重复即可。
    dag::Edge input(common::joinName(options["modelName"], "detect_in"));
    dag::Edge output(common::joinName(options["modelName"], "detect_out"));

    // 创建检测模型有向无环图graph
    dag::Graph *graph =
        dag::createGraph(options["modelName"], inference_type, device_type,
                         &input, &output, model_type, is_path, model_value);
    CHECK_GRAPH_NULLPTR(graph, "graph is nullptr");

    CHECK_GRAPH(graph->init(), "graph init failed");

    cv::Mat input_mat = cv::imread(options["inputPath"]);
    input.set(input_mat, 0);
    CHECK_GRAPH(graph->run(), "graph run failed");

    // 得到有向无环图graph的输出结果
    nndeploy::model::DetectResult *result =
        (nndeploy::model::DetectResult *)output.getParam(nullptr);
    CHECK_GRAPH_NULLPTR(result, "result is nullptr");

    util::drawBox(input_mat, *result);
    cv::imwrite(options["outputPath"], input_mat);

    // 有向无环图graphz反初始化
    CHECK_GRAPH(graph->deinit(), "graph deinit failed");

    // 有向无环图graphz销毁
    delete graph;

    return base::kStatusCodeOk;
  }
};
}  // namespace application
}  // namespace bevinfer

#endif /* _BEVINFER_APPLICATION_YOLOV8_PIPELINE_H_ */
