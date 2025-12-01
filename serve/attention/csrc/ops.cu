#include <torch/library.h>
#include <torch/version.h>

#include "batch_topk.cuh"

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def("batch_topk", &batch_topk);
}
