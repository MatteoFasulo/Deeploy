# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _Conv2D_Template(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        data_out = ctxt.lookup(operatorRepresentation['data_out'])

        operatorRepresentation['input_offset'] = 0
        if hasattr(data_in, "_signed") and hasattr(data_in, "nLevels"):
            operatorRepresentation['input_offset'] = (data_in._signed == 0) * int(data_in.nLevels // 2)
        operatorRepresentation['output_offset'] = 0
        if hasattr(data_out, "_signed") and hasattr(data_out, "nLevels"):
            operatorRepresentation['output_offset'] = -(data_out._signed == 0) * int(data_out.nLevels // 2)

        # Check if bias is available
        if 'bias' in operatorRepresentation:
            operatorRepresentation['has_bias'] = "true"
        else:
            operatorRepresentation['has_bias'] = "false"
            operatorRepresentation['bias'] = 'NULL'

        return ctxt, operatorRepresentation, []


reference2DTemplate = _Conv2D_Template("""
// 2D Int Conv HWC (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
    ${data_in_type.typeName} ref_${data_out}_${data_in} = ${data_in};
    ${data_out_type.typeName} ref_${data_out}_${data_out} = ${data_out};

    for (uint32_t n=0; n<${batch}; ++n) {
        Conv2d_s${data_in_type.referencedType.typeWidth}_s${weight_type.referencedType.typeWidth}_s${data_out_type.referencedType.typeWidth}_HWC(
            ref_${data_out}_${data_in}, ${dim_im_in_x}, ${dim_im_in_y}, ${ch_im_in},
            ${weight}, ${ch_im_out},
            ${dim_kernel_x}, ${dim_kernel_y},
            ${stride_x}, ${stride_y},
            ref_${data_out}_${data_out},
            ${input_offset}, ${output_offset},
            ${bias},
            ${padding_y_top}, ${padding_y_bottom}, ${padding_x_left}, ${padding_x_right}
        );
        ref_${data_out}_${data_in} += ${ch_im_in}*${dim_im_in_x}*${dim_im_in_y};
        ref_${data_out}_${data_out} += ${ch_im_out}*${dim_im_out_x}*${dim_im_out_y};
    }
END_SINGLE_CORE
""")
