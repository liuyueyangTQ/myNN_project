#pragma once
#include <cassert>
#include "network.pb.h"
#include "metrix.h"
#include "nn.h"
#include "enum_type.h"


void matrix2proto(const base::metrix_float& m, nn_proto::WeightMatrix& w_proto);

void bias2proto(const base::metrix_float& b_vec, nn_proto::BiasVector& b_proto);

void proto2matrix(const nn_proto::WeightMatrix& w_proto, base::metrix_float& m);

void proto2bias(const nn_proto::BiasVector& b_proto, base::metrix_float& b);

void network2proto(nn::module_base* net, nn_proto::Network& net_proto);

void proto2network(const nn_proto::Network& net_proto, nn::module_base* net);