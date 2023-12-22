#ifndef _BEVINFER_COMMON_INFER_GRAPH_UTIL_H_
#define _BEVINFER_COMMON_INFER_GRAPH_UTIL_H_
#include "common/inc/plog.h"
#include "nndeploy/base/status.h"

#define CHECK_GRAPH(status, str)                      \
  do {                                                \
    if (status != (nndeploy::base::kStatusCodeOk)) {  \
      ALOG_ERROR("%s\n", str);                        \
      return nndeploy::base::kStatusCodeErrorUnknown; \
    }                                                 \
  } while (0)

#define CHECK_GRAPH_NULLPTR(status, str)              \
  do {                                                \
    if (!status) {                                    \
      ALOG_ERROR("%s\n", str);                        \
      return nndeploy::base::kStatusCodeErrorUnknown; \
    }                                                 \
  } while (0)

#endif  // _BEVINFER_COMMON_INFER_GRAPH_UTIL_H_