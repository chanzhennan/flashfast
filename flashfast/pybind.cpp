#pragma once

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "ops/op.h"

PYBIND11_MODULE(flashfast, m) {
//////////////////////////////   ops   //////////////////////////////
    // ops: decode attention
    m.def("decode_mha_ut", &decode_mha_ut);
    // m.def("decode_mha_alibi_masked_ut", &decode_mha_alibi_masked_ut);
    // m.def("decode_mqa_ut", &decode_mqa_ut);
    // m.def("decode_mqa_t_ut", &decode_mqa_t_ut);
    // m.def("decode_serving_mha_ut", &decode_serving_mha_ut);
}
