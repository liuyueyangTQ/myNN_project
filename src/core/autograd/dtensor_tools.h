#pragma once
#include <vector>
#include "dtensor.h"
#include "enum_types.h"

// 用于定义 dtensor_factory 要创建的各种变量声明
#include "dtensors/dtensor_1D.h"
#include "dtensors/dtensor_2D.h"
#include "dtensors/dtensor_3D.h"
#include "dtensors/dtensor_common.h"

#include "dtensors/layers/layers.h" // 用于定义layer tools 的 sigmoid 之类的layer
namespace dtensor {

[[nodiscard]] dtensor_base* dtensor_factory(tensor_type p, std::vector<size_t>& tensor_shape, sub_type q, size_t batch_num, op* temp_op);

layer* layer_tool(int n, size_t batch_num, sub_type stp);

} // namespace dtensor