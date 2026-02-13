#include <orteaf/extension/ops/tensor_ops.h>

#include "detail/tensor_kind_ops.h"
#include "detail/tensor_kind_dispatch.h"

namespace orteaf::extension::ops {

namespace detail_ops = ::orteaf::extension::ops::detail;

void TensorOps::fill(Tensor &output, double value) {
  detail_ops::dispatch_out(output, "fill", [&]<typename KindTag>(KindTag) {
    using Impl = typename KindTag::type;
    detail_ops::kindFill<Impl>(output, value);
  });
}

void TensorOps::print(const Tensor &input) {
  detail_ops::dispatch_in(input, "print", [&]<typename KindTag>(KindTag) {
    using Impl = typename KindTag::type;
    detail_ops::kindPrint<Impl>(input);
  });
}

void TensorOps::copyHostToDevice(Tensor &output, const Tensor &input) {
  detail_ops::dispatch_out_in(
      output, input, "copyHostToDevice", [&]<typename KindTag>(KindTag) {
        using Impl = typename KindTag::type;
        detail_ops::kindCopyHostToDevice<Impl>(output, input);
      });
}

void TensorOps::copyDeviceToHost(Tensor &output, const Tensor &input) {
  detail_ops::dispatch_out_in(
      output, input, "copyDeviceToHost", [&]<typename KindTag>(KindTag) {
        using Impl = typename KindTag::type;
        detail_ops::kindCopyDeviceToHost<Impl>(output, input);
      });
}

} // namespace orteaf::extension::ops
