#ifndef _BEVINFER_COMMON_INFER_GRAPH_H_
#define _BEVINFER_COMMON_INFER_GRAPH_H_

#include "nndeploy/base/status.h"

namespace bevinfer {
namespace infer {
using namespace nndeploy;
class ApplicationBaseInfer {
 public:
  virtual base::Status run() = 0;
};
}  // namespace infer
}  // namespace bevinfer
#endif  // _BEVINFER_COMMON_INFER_GRAPH_H_