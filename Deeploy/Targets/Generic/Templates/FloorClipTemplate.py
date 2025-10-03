# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate


class _FloorClipTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)


referenceTemplate = _FloorClipTemplate("""
// FloorClip (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE

    for (uint32_t i = 0; i < ${size}; i++) {
        ${data_in_type.referencedType.typeName} input_val = ${data_in}[i];
        ${data_in_type.referencedType.typeName} floored_val = floor(input_val);
        
        // Apply clipping
        if (floored_val < ${min_val}) floored_val = ${min_val};
        if (floored_val > ${max_val}) floored_val = ${max_val};

        ${data_out}[i] = (${data_out_type.referencedType.typeName})(floored_val);
    }

END_SINGLE_CORE
""")