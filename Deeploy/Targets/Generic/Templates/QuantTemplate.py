# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate


class _QuantTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)


referenceTemplate = _QuantTemplate("""
// Quantization (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE

    for (uint32_t i = 0; i < ${size}; i++) {

        float32_t input_val = ${data_in}[i];
        float32_t inv_scale = 1.0 / (float32_t)${scale};
        float32_t scaled_val  = (float32_t)input_val / inv_scale;
        float32_t shifted_val = scaled_val + (float32_t)${zero_point};

        int32_t quantized = (int32_t)floor(shifted_val + 0.5);

        if (quantized < ${min_val}) quantized = ${min_val};
        if (quantized > ${max_val}) quantized = ${max_val};

        ${data_out}[i] = (${data_out_type.referencedType.typeName})(quantized);
    }

END_SINGLE_CORE
""")
