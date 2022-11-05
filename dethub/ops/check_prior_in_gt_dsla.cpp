#include "pytorch_cpp_helper.hpp"

void CheckPriorsInGtDSLALauncher(Tensor priors, Tensor gt_bboxes,
                                 Tensor is_in_gts);

void check_prior_in_gt_dsla(Tensor priors, Tensor gt_bboxes, Tensor is_in_gts) {
  CHECK_CUDA_INPUT(priors);
  CHECK_CUDA_INPUT(gt_bboxes);
  CHECK_CUDA_INPUT(is_in_gts);

  return CheckPriorsInGtDSLALauncher(priors, gt_bboxes, is_in_gts);
}
