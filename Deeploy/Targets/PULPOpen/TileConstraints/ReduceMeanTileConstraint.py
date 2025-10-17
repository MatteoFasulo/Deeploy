# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple, Union

from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme


class ReduceMeanTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        """
        Adds the geometrical constraints for the ReduceMean operation to the TilerModel.
        This function correctly handles the 'keepdims' attribute to define the relationship
        between input and output tensor shapes.
        """
        # Get tensor buffer names
        inputBufferName = parseDict['data_in']
        outputBufferName = parseDict['data_out']

        # Add I/O dimensions to the model as variables
        for bufferName in [inputBufferName, outputBufferName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        input_shape = ctxt.lookup(inputBufferName).shape
        output_shape = ctxt.lookup(outputBufferName).shape

        axes = parseDict['axes']
        keepdims = parseDict.get('keepdims', 1)

        if keepdims:
            # If keepdims is True, ranks are the same.
            assert len(input_shape) == len(output_shape)
            for i in range(len(input_shape)):
                inputDimVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = i)
                outputDimVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = i)
                if i in axes:
                    tilerModel.addConstraint(outputDimVar == 1)
                else:
                    tilerModel.addConstraint(inputDimVar == outputDimVar)
        else:
            # If keepdims is False, output rank is smaller.
            assert len(output_shape) == len(input_shape) - len(axes)
            output_idx = 0
            for i in range(len(input_shape)):
                inputDimVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = i)
                if i not in axes:
                    # Map the non-reduced input dimension to the corresponding output dimension.
                    outputDimVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = output_idx)
                    tilerModel.addConstraint(inputDimVar == outputDimVar)
                    output_idx += 1

        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        """
        Adds policy constraints for the ReduceMean operation.
        Enforces that channel tile size * reduction_len * elem_size_bytes <= MAX_DMA_BYTES
        so any single DMA remains representable by the 17-bit Mchan limit.
        """
        # safe constant for Mchan transfer size
        MAX_DMA_BYTES = (1 << 17) - 1  # 131071

        # element size in bytes: try parseDict/operatorRepresentation first, fallback to float32
        elem_size = int(parseDict.get('elem_size_bytes', 4))  # default 4 bytes (float32)

        # input tensor
        input_buf = ctxt.lookup(name = parseDict['data_in'])
        input_shape = tuple(input_buf.shape)  # may be (1,25,3072)
        rank = len(input_shape)

        # axes: allow parseDict to supply a single int, list, tuple or numpy array
        raw_axes = parseDict.get('axes', None)
        if raw_axes is None:
            # default: reduce all axes (rare in your case)
            reduced_axes = set(range(rank))
        else:
            # normalize axes to a set of ints
            if isinstance(raw_axes, (list, tuple)):
                axes_list = list(raw_axes)
            else:
                try:
                    axes_list = list(raw_axes)
                except Exception:
                    axes_list = [int(raw_axes)]
            reduced_axes = set([int(a) % rank for a in axes_list])

        # choose a "channel" axis among non-reduced axes:
        non_reduced = [i for i in range(rank) if i not in reduced_axes]
        if not non_reduced:
            # Nothing to tile: all axes reduced -> no channel axis
            return tilerModel

        # Heuristic: pick the largest non-reduced dimension as channel axis
        channel_axis = max(non_reduced, key = lambda i: int(input_shape[i]))

        # Get the TilerModel var for that input dim (we constrain the tile size of the input dim)
        # Note: tile dim var uses tensor name and dimIdx relative to the tensor; using input tensor is safe
        channelVar = tilerModel.getTensorDimVar(tensorName = input_buf.name, dimIdx = channel_axis)

        # reduction length (number of elements along reduced axes) — conservative approach:
        # if there are multiple reduced axes, multiply lengths; in your case it's a single axis length 25
        red_len = 1
        for r in sorted(reduced_axes):
            red_len *= int(input_shape[r])

        # Now add the linear constraint:
        # channelVar * (elem_size * red_len) <= MAX_DMA_BYTES
        # (i.e. channelVar <= floor(MAX_DMA_BYTES / (elem_size*red_len)))
        tilerModel.addConstraint((channelVar * (elem_size * red_len)) <= MAX_DMA_BYTES)

        # Optionally add a conservative upper limit (e.g., <=1024) to avoid extreme tiles
        # Uncomment if you want an extra safe bound:
        # tilerModel.addConstraint(channelVar <= 1024)

        return tilerModel

    @staticmethod
    def constructSymbolicNodeRep(tilerModel: TilerModel, parseDict: Dict,
                                 ctxt: NetworkContext) -> Dict[str, Union[int, IntVar]]:
        return parseDict.copy()

    @staticmethod
    def computeInputTile(outputCube: HyperRectangle, input_shape: Tuple[int, ...], axes: List[int]) -> HyperRectangle:
        """
        Computes the required input tile for a given output tile.
        """
        rank = len(input_shape)
        input_offsets = [0] * rank
        input_dims = list(input_shape)

        output_dims = list(outputCube.dims)
        output_offsets = list(outputCube.offset)

        # Determine mapping mode: keepdims vs squeezed output
        if len(output_dims) == rank:
            # keepdims=True mapping: output has a slot for every input dim (reduced dims are size 1)
            for i in range(rank):
                if i not in axes:
                    # take offset/dim from same index i
                    input_offsets[i] = output_offsets[i]
                    input_dims[i] = output_dims[i]
                else:
                    # reduced axis: full input dimension and offset 0 (or whatever policy)
                    input_offsets[i] = 0
                    input_dims[i] = input_shape[i]
        elif len(output_dims) == rank - len(axes):
            # keepdims=False mapping: output is squeezed, iterate output indices
            out_idx = 0
            for i in range(rank):
                if i not in axes:
                    input_offsets[i] = output_offsets[out_idx]
                    input_dims[i] = output_dims[out_idx]
                    out_idx += 1
                else:
                    input_offsets[i] = 0
                    input_dims[i] = input_shape[i]
        else:
            # Mismatched shapes — fail early with informative message
            raise ValueError(f"Output cube rank ({len(output_dims)}) doesn't match expected "
                             f"keepdims layouts for input rank {rank} and axes {axes}.")

        return HyperRectangle(tuple(input_offsets), tuple(input_dims))

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        addrNames = ['data_in', 'data_out']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        input_buffer = ctxt.lookup(operatorRepresentation['data_in'])
        input_shape = input_buffer.shape
        axes = operatorRepresentation['axes']

        # Compute the corresponding input cube for each output tile cube.
        inputInCubes = [cls.computeInputTile(cube, input_shape, axes) for cube in outputCubes]

        # Create the load schedules for the TilingSchedule object.
        inputLoadSchedule = [{"data_in": in_cube} for in_cube in inputInCubes]
        outputLoadSchedule = [{"data_out": out_cube} for out_cube in outputCubes]

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)

        variableReplacementScheme = VariableReplacementScheme({}, {})

        return variableReplacementScheme, tilingSchedule
