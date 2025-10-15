# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

from Deeploy.DeeployTypes import NodeTemplate, OperatorRepresentation


class ClipTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: Dict, operatorRepresentation: OperatorRepresentation, **kwargs):

        data_in = ctxt['data_in']
        data_out = ctxt['data_out']

        size = operatorRepresentation['size']
        min_val = operatorRepresentation.get('min_val', -128)  # FBRANCASI: Default for int8
        max_val = operatorRepresentation.get('max_val', 127)  # FBRANCASI: Default for int8

        newCtxt = {
            'data_in': data_in,
            'data_out': data_out,
            'size': size,
            'min_val': min_val,
            'max_val': max_val,
        }

        return newCtxt


referenceTemplate = ClipTemplate("""
for(int i = 0; i < ${size}; i++) {
    ${data_in_type.referencedType.typeName} val = ${data_in}[i];
    if (val < ${min_val}) {
        ${data_out}[i] = (${data_in_type.referencedType.typeName})${min_val};
    } else if (val > ${max_val}) {
        ${data_out}[i] = (${data_in_type.referencedType.typeName})${max_val};
    } else {
        ${data_out}[i] = val;
    }
}
""")