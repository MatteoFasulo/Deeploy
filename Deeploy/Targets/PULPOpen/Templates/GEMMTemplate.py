# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class PULPGEMMTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        # Extract signedness information of input, weights and output
        signedW = ctxt.lookup(operatorRepresentation['B'])._type.referencedType.typeMin < 0
        signedI = ctxt.lookup(operatorRepresentation['A'])._type.referencedType.typeMin < 0
        signedO = ctxt.lookup(operatorRepresentation['data_out'])._type.referencedType.typeMin < 0
        operatorRepresentation['weight_signed'] = signedW
        operatorRepresentation['input_signed'] = signedI
        operatorRepresentation['output_signed'] = signedO

        return ctxt, operatorRepresentation, []


PULPGEMM_8_Template = PULPGEMMTemplate("""
<%
signatureString = ''
if input_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
if output_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
if weight_signed:
    signatureString += '_i8'
else:
    signatureString += '_u8'
%>
// PULP NN GEMM - Keep original for now
int8_t* ref_${data_out}_${A} = ${A};
int8_t* ref_${data_out}_${B} = ${B};
int8_t* ref_${data_out}_${data_out} = ${data_out};
for(int i=0;i<${batch};i++){
for(int j=0;j<${M};j++){
// LMACAN: In some edge cases sporadic errors happen if this loop is not added.
// We believe this is due to missing bubbles in the pipeline that break operator forwarding.
// Breaking test:
//   `python testRunner_tiled_siracusa.py -t=Tests/Transformer --defaultMemLevel=L3 --doublebuffer --l1=30000`
#pragma unroll 1
for(int k=0;k<3;k++){
  asm volatile("nop" ::);
}
pulp_nn_linear${signatureString}(ref_${data_out}_${A}, NULL, ref_${data_out}_${data_out}, ref_${data_out}_${B}, ${mul}, ${C}, 1, ${log2D}, ${N}, ${O}, 1, 1);
ref_${data_out}_${A} += ${N};
ref_${data_out}_${data_out} += ${O};
}
% if W_batched:
ref_${data_out}_${B} += ${N} * ${O};
% endif
}
""")


class _MatMulTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        A = ctxt.lookup(operatorRepresentation['A'])
        B = ctxt.lookup(operatorRepresentation['B'])
        C = ctxt.lookup(operatorRepresentation['data_out'])

        operatorRepresentation['A_offset'] = 0
        operatorRepresentation['B_offset'] = 0
        operatorRepresentation['C_offset'] = 0

        if hasattr(A, "nLevels"):
            operatorRepresentation['A_offset'] = (A._type.referencedType.typeMin == 0) * int(A.nLevels / 2)
        if hasattr(B, "nLevels"):
            operatorRepresentation['B_offset'] = (B._type.referencedType.typeMin == 0) * int(B.nLevels / 2)
        if hasattr(C, "nLevels"):
            operatorRepresentation['C_offset'] = -(C._type.referencedType.typeMin == 0) * int(C.nLevels / 2)

        return ctxt, operatorRepresentation, []


PULPMM_8_Template = _MatMulTemplate("""
// Direct MatMul call (Name: ${nodeName}, Op: ${nodeOp})
${A_type.typeName} ref_${data_out}_${A} = ${A};
${B_type.typeName} ref_${data_out}_${B} = ${B};
${data_out_type.typeName} ref_${data_out}_${data_out} = ${data_out};

for(uint32_t i=0;i<${batch};i++){
    pi_cl_team_fork(NUM_CORES, PULP_MatMul_s${A_type.referencedType.typeWidth}_s${B_type.referencedType.typeWidth}_s${data_out_type.referencedType.typeWidth}_unroll1x7, &(struct {
        const ${A_type.referencedType.typeName} *pSrcA;
        const ${B_type.referencedType.typeName} *pSrcB;
        ${data_out_type.referencedType.typeName} *pDstY;
        uint32_t M, N, O;
    }){
        .pSrcA = ref_${data_out}_${A},
        .pSrcB = ref_${data_out}_${B},
        .pDstY = ref_${data_out}_${data_out},
        .M = ${M},
        .N = ${N},
        .O = ${O}
    });

    ref_${data_out}_${A} += ${M} * ${N};
    ref_${data_out}_${B} += ${N} * ${O};
    ref_${data_out}_${data_out} += ${M} * ${O};
}
""")