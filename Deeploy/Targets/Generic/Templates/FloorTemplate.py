# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

from Deeploy.DeeployTypes import NodeTemplate, OperatorRepresentation


class FloorTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: Dict, operatorRepresentation: OperatorRepresentation, **kwargs):

        data_in = ctxt['data_in']
        data_out = ctxt['data_out']

        size = operatorRepresentation['size']

        newCtxt = {
            'data_in': data_in,
            'data_out': data_out,
            'size': size,
        }

        return newCtxt


referenceTemplate = FloorTemplate("""
for(int i = 0; i < ${size}; i++) {
    // Apply floor operation, preserving the data type
    ${data_out}[i] = (${data_in_type.referencedType.typeName})floorf((float)${data_in}[i]);
}
""")