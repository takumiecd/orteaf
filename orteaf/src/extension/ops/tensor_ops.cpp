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

void TensorOps::copyHostToMps(Tensor &output, const Tensor &input) {
  detail_ops::dispatch_out_in(
      output, input, "copyHostToMps", [&]<typename KindTag>(KindTag) {
        using Impl = typename KindTag::type;
        detail_ops::kindCopyHostToMps<Impl>(output, input);
      });
}

void TensorOps::copyMpsToHost(Tensor &output, const Tensor &input) {
  detail_ops::dispatch_out_in(
      output, input, "copyMpsToHost", [&]<typename KindTag>(KindTag) {
        using Impl = typename KindTag::type;
        detail_ops::kindCopyMpsToHost<Impl>(output, input);
      });
}

} // namespace orteaf::extension::ops
