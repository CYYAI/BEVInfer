#include "yolov8.h"

namespace bevinfer {
namespace application {

// 注册该模型推理
dag::TypeGraphRegister g_register_yolov8_graph(BEVINFER_YOLOV8,
                                               createYoloV8Graph);

// 2. 实现前处理的run方法，其实就是你的模型输入的前处理
base::Status YoloPreProcess::run() {
  YoloPreParam *yolov8_pre_param = dynamic_cast<YoloPreParam *>(param_.get());

  cv::Mat *src = inputs_[0]->getCvMat(this);

  // 你自己创建所需的tensor，如果是动态batch/宽高,方便直接输入
  device::Device *device = device::getDefaultHostDevice();
  device::TensorDesc desc;
  desc.data_type_ = base::dataTypeOf<float>();
  desc.data_format_ = base::kDataFormatNCHW;
  desc.shape_.emplace_back(1);
  desc.shape_.emplace_back(
      model::getChannelByPixelType(yolov8_pre_param->dst_pixel_type_));
  desc.shape_.emplace_back(yolov8_pre_param->h_);
  desc.shape_.emplace_back(yolov8_pre_param->w_);
  outputs_[0]->create(device, desc, inputs_[0]->getIndex(this));
  device::Tensor *dst = outputs_[0]->getTensor(this);

  std::string name = dst->getName();
  int c = dst->getChannel();
  int h = dst->getHeight();
  int w = dst->getWidth();

  cv::Mat tmp_cvt;
  if (yolov8_pre_param->src_pixel_type_ != yolov8_pre_param->dst_pixel_type_) {
    base::CvtColorType cvt_type = base::calCvtColorType(
        yolov8_pre_param->src_pixel_type_, yolov8_pre_param->dst_pixel_type_);
    if (cvt_type == base::kCvtColorTypeNotSupport) {
      NNDEPLOY_LOGE("cvtColor type not support");
      return base::kStatusCodeErrorNotSupport;
    }
    int cv_cvt_type = model::OpenCvConvert::convertFromCvtColorType(cvt_type);
    cv::cvtColor(*src, tmp_cvt, cv_cvt_type);
  } else {
    tmp_cvt = *src;
  }

  cv::Mat tmp_resize;
  if (yolov8_pre_param->interp_type_ != base::kInterpTypeNotSupport) {
    int interp_type = model::OpenCvConvert::convertFromInterpType(
        yolov8_pre_param->interp_type_);
    cv::resize(tmp_cvt, tmp_resize, cv::Size(w, h), 0.0, 0.0, interp_type);
  } else {
    tmp_resize = tmp_cvt;
  }

  model::OpenCvConvert::convertToTensor(
      tmp_resize, dst, yolov8_pre_param->scale_, yolov8_pre_param->mean_,
      yolov8_pre_param->std_);
  return base::kStatusCodeOk;
}

base::Status YoloPostProcess::run() {
  YoloPostParam *param = (YoloPostParam *)param_.get();
  float score_threshold = param->score_threshold_;
  int num_classes = param->num_classes_;

  // 单个输出结果的获取
  device::Tensor *tensor = inputs_[0]->getTensor(this);
  float *data = (float *)tensor->getPtr();
  int batch = tensor->getShapeIndex(0);
  int height = tensor->getShapeIndex(1);
  int width = tensor->getShapeIndex(2);
  // 多个输出结果的获取
  // device::Tensor *tensor_stride_8 = inputs_[0]->getTensor(this);
  // device::Tensor *tensor_stride_16 = inputs_[1]->getTensor(this);
  // device::Tensor *tensor_stride_32 = inputs_[2]->getTensor(this);

  cv::Mat cv_mat_src(height, width, CV_32FC1, data);
  cv::Mat cv_mat_dst(width, height, CV_32FC1);
  cv::transpose(cv_mat_src, cv_mat_dst);
  std::swap(height, width);
  data = (float *)cv_mat_dst.data;

  nndeploy::model::DetectResult *results = new nndeploy::model::DetectResult();
  outputs_[0]->set(results, inputs_[0]->getIndex(this), false);

  for (int b = 0; b < batch; ++b) {
    float *data_batch = data + b * height * width;
    nndeploy::model::DetectResult results_batch;
    for (int h = 0; h < height; ++h) {
      float *data_row = data_batch + h * width;
      float x_center = data_row[0];
      float y_center = data_row[1];
      float object_w = data_row[2];
      float object_h = data_row[3];
      float x0 = x_center - object_w * 0.5f;
      x0 = x0 > 0.0 ? x0 : 0.0;
      float y0 = y_center - object_h * 0.5f;
      y0 = y0 > 0.0 ? y0 : 0.0;
      float x1 = x_center + object_w * 0.5f;
      x1 = x1 < param->model_w_ ? x1 : param->model_w_;
      float y1 = y_center + object_h * 0.5f;
      y1 = y1 < param->model_h_ ? y1 : param->model_h_;
      for (int class_idx = 0; class_idx < num_classes; ++class_idx) {
        float score = data_row[4 + class_idx];
        if (score > score_threshold) {
          nndeploy::model::DetectBBoxResult bbox;
          bbox.index_ = b;
          bbox.label_id_ = class_idx;
          bbox.score_ = score;
          bbox.bbox_[0] = x0;
          bbox.bbox_[1] = y0;
          bbox.bbox_[2] = x1;
          bbox.bbox_[3] = y1;
          results_batch.bboxs_.emplace_back(bbox);
        }
      }
    }
    std::vector<int> keep_idxs(results_batch.bboxs_.size());
    nndeploy::model::computeNMS(results_batch, keep_idxs,
                                param->nms_threshold_);
    for (auto i = 0; i < keep_idxs.size(); ++i) {
      auto n = keep_idxs[i];
      if (n < 0) {
        continue;
      }
      results_batch.bboxs_[n].bbox_[0] /= param->model_w_;
      results_batch.bboxs_[n].bbox_[1] /= param->model_h_;
      results_batch.bboxs_[n].bbox_[2] /= param->model_w_;
      results_batch.bboxs_[n].bbox_[3] /= param->model_h_;
      results->bboxs_.emplace_back(results_batch.bboxs_[n]);
    }
  }

  return base::kStatusCodeOk;
}

dag::Graph *createYoloV8Graph(const std::string &name,
                              base::InferenceType inference_type,
                              base::DeviceType device_type, dag::Edge *input,
                              dag::Edge *output, base::ModelType model_type,
                              bool is_path,
                              std::vector<std::string> model_value) {
  dag::Graph *graph = new dag::Graph(name, input, output);

  // 下面是单个输入输出的例子，注意，input/output的name就是你onnx中的名称，需要自己指定
  dag::Edge *infer_input = graph->createEdge("images");
  dag::Edge *infer_output = graph->createEdge("output0");

  dag::Node *pre = graph->createNode<YoloPreProcess>(
      common::joinName(name, "preprocess"), input, infer_input);

  dag::Node *infer = graph->createInfer<model::Infer>(
      common::joinName(name, "infer"), inference_type, infer_input,
      infer_output);

  dag::Node *post = graph->createNode<YoloPostProcess>(
      common::joinName(name, "postprocess"), infer_output, output);

  // 如果有多个输入输出，需要按照下面格式写就行，输入输出名称需自己指定
  // dag::Edge *infer_input = graph->createEdge("infer_input");
  // dag::Edge *edge_stride_8 = graph->createEdge("output");  // [1, 3, 80, 80,
  // 85] dag::Edge *edge_stride_16 = graph->createEdge("376");    // [1, 3, 40,
  // 40, 85] dag::Edge *edge_stride_32 = graph->createEdge("401");    // [1, 3,
  // 20, 20, 85]

  // dag::Node *pre = graph->createNode<YoloPreProcess>("preprocess", input,
  //                                                           infer_input);

  // dag::Node *infer = graph->createInfer<model::Infer>(
  //     "infer", inference_type, {infer_input},
  //     {edge_stride_8, edge_stride_16, edge_stride_32});

  // dag::Node *post = graph->createNode<YoloMultiOutputPostProcess>(
  //     "postprocess", {edge_stride_8, edge_stride_16, edge_stride_32},
  //     {output});

  inference::InferenceParam *inference_param =
      (inference::InferenceParam *)(infer->getParam());
  inference_param->is_path_ = is_path;
  inference_param->model_value_ = model_value;
  inference_param->device_type_ = device_type;
  inference_param->model_type_ = model_type;

  return graph;
}

}  // namespace application
}  // namespace bevinfer
