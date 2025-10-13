# ----------------------------------------------------------------------
#
# File: IntConvTemplate.py
#
# Last edited: 23.01.2025
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author: Federico Brancasi, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Tuple, Union

from ortools.constraint_solver.pywrapcp import IntVar

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


class PULP2DIntConvIm2ColTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    @staticmethod
    def computeTransientBuffersSize(
            ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> List[Tuple[str, Union[int, IntVar]]]:
        im2col_dim = 8 * (operatorRepresentation['ch_im_in'] * operatorRepresentation['dim_kernel_x'] *
                          operatorRepresentation['dim_kernel_y'])
        im2col_name = operatorRepresentation['nodeName'] + "_buffer"
        return [(im2col_name, im2col_dim)]

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

    def hoistTransientBuffers(self, ctxt: NetworkContext,
                              operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        im2col_name, im2col_dim = PULP2DIntConvIm2ColTemplate.computeTransientBuffersSize(ctxt,
                                                                                          operatorRepresentation)[0]
        ctxt.hoistTransientBuffer(im2col_name, im2col_dim)

        operatorRepresentation['ctxtBuffer'] = im2col_name
        operatorRepresentation['ctxtBufferSize'] = im2col_dim
        return ctxt, operatorRepresentation, [im2col_name]


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

reference2DIm2ColTemplate = PULP2DIntConvIm2ColTemplate("""
// 2D Int Conv HWC with Im2Col and multi-core parallelism (Name: ${nodeName}, Op: ${nodeOp})

${data_in_type.typeName} ref_${data_out}_${data_in} = ${data_in};
${data_out_type.typeName} ref_${data_out}_${data_out} = ${data_out};

for (uint32_t n=0; n<${batch}; ++n) {
    PULP_Conv2d_Im2Col_s${data_in_type.referencedType.typeWidth}_s${weight_type.referencedType.typeWidth}_s${data_out_type.referencedType.typeWidth}_HWC(
        ref_${data_out}_${data_in},
        ${dim_im_in_y},
        ${dim_im_in_x},
        ${ch_im_in},
        ${weight},
        ${ch_im_out},
        ${dim_kernel_y},
        ${dim_kernel_x},
        ${stride_y},
        ${stride_x},
        ref_${data_out}_${data_out},
        ${input_offset},
        ${output_offset},
        ${bias},
        ${padding_y_top},
        ${padding_y_bottom},
        ${padding_x_left},
        ${padding_x_right},
        ${ctxtBuffer}
    );

    ref_${data_out}_${data_in} += ${ch_im_in} * ${dim_im_in_x} * ${dim_im_in_y};
    ref_${data_out}_${data_out} += ${ch_im_out} * ${dim_im_out_x} * ${dim_im_out_y};
}
""")
