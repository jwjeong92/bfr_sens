import torch
from lib.gptq.gptq import GPTQ
from lib.quantization.quantizer import Quantizer, quantize, quant_int, dequant_int

from .weight_permutation import get_permutation_order
from typing import NamedTuple, Optional, Union

class QuantizationResult(NamedTuple):
    weight: torch.FloatTensor  # dequantized(quantized(weight)), same shape as the original
    perm: Optional[torch.LongTensor]  # optional input permutation indices that were used during quantization
    # NOTE: if permutation_order != identity, all subsequent tensors (incl. outlier indices) are permuted in that order!
    quantization_errors: torch.Tensor  # per-element quantization errors, defined as (weight - quantized_weight) / diag(inverse_hessan_cholesky)
    unstructured_outlier_threshold: float  # threshold on squared error increase used for determining *UNSTRUCTURED* outliers
    unstructured_outlier_mask: torch.Tensor  # bool mask where True means that this is an individual outlier
    save_quant_dict: dict


def get_leave_one_out_error(group_weight: torch.Tensor, group_diag_hessian_inv_cho: torch.Tensor, *, bits, sym):
    """EXPERIMENTAL! BEWARE - for each weight, fit quantizer without this_one_weight and return this one weight's reconstruction"""

    assert group_weight.ndim == 2
    loo_indices = torch.arange(group_weight.shape[1], device=group_weight.device)
    loo_indices = loo_indices[1:] - (loo_indices[:, None] >= loo_indices[1:]).to(loo_indices.dtype)
    groupwise_loo_data = group_weight[:, loo_indices]  # [num_groups, num_loo = groupsize, groupsize - 1]
    fast_quantizer = Quantizer(shape=groupwise_loo_data.flatten(0, 1).shape)
    fast_quantizer.configure(bits, perchannel=True, sym=sym)
    fast_quantizer.find_params(groupwise_loo_data.flatten(0, 1), weight=True)

    # compute error improvement from not quantizing each one weight
    # to do so, we shall first train quantizer on leave-one-out data (which can be done faster since not all data affects quantization)
    loo_groupwise_reconstructed_weights = fast_quantizer.quantize(
        groupwise_loo_data.flatten(0, 1)
    ).reshape_as(groupwise_loo_data)
    loo_group_diag_hessian_inv_cho = group_diag_hessian_inv_cho[loo_indices]  # [num_loo = groupsize, groupsize - 1]
    assert group_diag_hessian_inv_cho.ndim == 1

    # total quantization error consists of hessian-weighted mse on all remaining weights except for the one that's left out
    # -- this is because the left-out weights will not be quantized, and therefore, has zero quantization error
    loo_errors_sq = (
        ((loo_groupwise_reconstructed_weights - groupwise_loo_data) / loo_group_diag_hessian_inv_cho).square().sum(-1)
    )
    assert loo_errors_sq.shape == group_weight.shape  # [num_groups, num_loo = groupsize]

    # as a baseline error, quantize data normally without outliers
    base_quantizer = Quantizer(shape=group_weight.shape)
    base_quantizer.configure(bits, perchannel=True, sym=sym)
    base_quantizer.find_params(group_weight, weight=True)
    baseline_reconstructed_weights = base_quantizer.quantize(group_weight)
    baseline_errors_sq = (
        ((baseline_reconstructed_weights - group_weight) / group_diag_hessian_inv_cho).square().sum(dim=1, keepdim=True)
    )

    # outlier's usefulness = how much does mse decrease from treating this weight as an outlier
    reduction_in_squared_error = baseline_errors_sq - loo_errors_sq
    return reduction_in_squared_error


class SPQR(GPTQ): # init, add_batch -> re-use
    def __init__(self, layer):
        super().__init__(self, layer)

    def add_batch(self, inp, out):
        super().add_batch(self, inp, out)

    def layer_quant(
        self,
        *,
        bits: int = 2,
        blocksize: int = 128,
        percdamp: float = 1e-2,
        groupsize: Optional[int] = None,
        keep_last_columns: int = 0,
        outlier_relative_threshold: float = float("inf"),
        permutation_order: Union[str, torch.Tensor] = "identity",
        keep_H: bool = True,
        simplified_outliers: bool = False,
        verbose=True,
        perchannel: bool = True,
        sym: bool = False,
        save_quantization: bool = False,
        **kwargs,
    ) -> QuantizationResult:
        
        weight = self.layer.weight.detach().to(dtype=torch.float, copy=True)
        save_quant_dict = {}
        perm = get_permutation_order(self.H, weight, permutation_order)

        if save_quantization:
            save_quant_dict["quant_weights"] = []
            save_quant_dict["quant_layer_scale"] = []
            save_quant_dict["quant_layer_zeros"] = []
            save_quant_dict["quant_layer_scale_qq_scale"] = []
            save_quant_dict["quant_layer_scale_qq_zero"] = []
            save_quant_dict["quant_layer_zero_qq_scale"] = []
            save_quant_dict["quant_layer_zero_qq_zero"] = []
            save_quant_dict["save_float_dtype"] = self.layer.weight.dtype
            save_quant_dict["outliers_matrix"] = torch.zeros(
                weight.shape, dtype=save_quant_dict["save_float_dtype"]
            ).to(
                weight.device
            )  # shape = [out_features, in_features]

        weight = weight[:, perm]
        H = self.H
        if keep_H:
            H = H.clone()
        else:
            self.H = None

        H = H[perm][:, perm]
        self.dead = torch.diag(H) == 0
        if percdamp > 0:
            ix = torch.arange(len(H), device=weight.device)
            H[ix, ix] += percdamp * abs(torch.diag(H)).mean()
            del ix
        H[self.dead, self.dead] = 1
        weight[:, self.dead] = 0
        H_inv = torch.cholesky_inverse(torch.linalg.cholesky(H))
        H_inv_cho = torch.linalg.cholesky(H_inv, upper=True)
        H_inv_cho_diag = torch.diag(H_inv_cho)
        del H

        quantizer = Quantizer()
        quantizer.configure(bits, perchannel=perchannel, sym=sym, **kwargs)
        assert H_inv_cho.shape[0] == H_inv_cho.shape[1] == weight.shape[1], "weight must be [out_features, in_features]"
        del H_inv

        out_dim, in_dim = weight.shape  # [out_features, in_features]

        if groupsize is None:
            groupsize = in_dim

        # prepare outlier detection
        outlier_column_indices = torch.empty(0, dtype=torch.int64, device=weight.device)
        
        outlier_scale = (weight.var(dim=0) / torch.diag(H_inv_cho).square()).mean().item()
        unstructured_outlier_threshold = outlier_relative_threshold * outlier_scale
        in_group_index = -1  # index of current group of input features, for group quantizer purposes

        quantization_errors = torch.zeros_like(weight)
        unstructured_outlier_mask = torch.zeros_like(weight, dtype=torch.bool)

        block_start_iter = range(0, in_dim - keep_last_columns, blocksize)
        for block_start in block_start_iter:
            block_end = min(block_start + blocksize, in_dim)
            for column_index in range(block_start, block_end):
                if column_index % groupsize == 0:
                    in_group_index += 1
                    group_weight = weight[:, column_index : column_index + groupsize]

                    if simplified_outliers or (unstructured_outlier_threshold == float("inf")):  # without outliers
                        quantizer.find_params(group_weight, weight=True)
                    
                    else:
                        # objective: detect which weights will be designated as outliers, fit quantizer without these weights
                        # step 1: fit quantizer on a elave-one-out vesion of weights, i.e. in each group, drop one weight at a time
                        assert perchannel, "refitting quatizer is only implemented for perchannel=True"
                        group_diag_hessian_inv_cho = H_inv_cho_diag[column_index : column_index + groupsize]
                        loo_quantization_error_sq = get_leave_one_out_error(
                            group_weight, group_diag_hessian_inv_cho, bits=bits, sym=sym
                        )
                        # ^-- dequantized(quantized(group_weight)) using a quantizer trained on all weights except the reconstructed one

                        likely_unstructued_outlier_mask = (
                            loo_quantization_error_sq > unstructured_outlier_threshold
                        ).float()

                        non_outlier_mask = 1 - likely_unstructued_outlier_mask
                        mean_over_non_outliers = torch.sum(
                            group_weight * non_outlier_mask, dim=1, keepdim=True
                        ) / torch.sum(non_outlier_mask, dim=1, keepdim=True).clamp_min(1)
                        group_weight_without_outliers = group_weight * non_outlier_mask + mean_over_non_outliers * (
                            1 - non_outlier_mask
                        )
                        quantizer.find_params(group_weight_without_outliers, weight=True)
                        del group_diag_hessian_inv_cho, loo_quantization_error_sq
                        del mean_over_non_outliers, group_weight_without_outliers, non_outlier_mask
                    
                    if save_quantization:
                        if quantizer.qq_scale_bits is not None:
                            save_quant_dict["quant_layer_scale"].append(quantizer.quant_scale.to(torch.int8))
                            save_quant_dict["quant_layer_scale_qq_scale"].append(
                                quantizer.qq_scale.scale.to(save_quant_dict["save_float_dtype"])
                            )
                            save_quant_dict["quant_layer_scale_qq_zero"].append(
                                quantizer.qq_scale.zero.to(save_quant_dict["save_float_dtype"])
                            )
                        else:
                            save_quant_dict["quant_layer_scale"].append(
                                quantizer.scale.to(save_quant_dict["save_float_dtype"])
                            )
                        
                        if quantizer.qq_zero_bits is not None and (
                            (not quantizer.round_zero) or quantizer.qq_zero_bits < quantizer.bits
                        ):
                            save_quant_dict["quant_layer_zeros"].append(quantizer.quant_zero.to(torch.int8))
                            save_quant_dict["quant_layer_zero_qq_scale"].append(
                                quantizer.qq_zero.scale.to(save_quant_dict["save_float_dtype"])
                            )
                            save_quant_dict["quant_layer_zero_qq_zero"].append(
                                quantizer.qq_zero.zero.to(save_quant_dict["save_float_dtype"])
                            )
                        else:
                            save_quant_dict["quant_layer_zeros"].append(
                                quantizer.zero.to(save_quant_dict["save_float_dtype"])
                            )
                    del group_weight
                    # Remove unstructured outliers in group_weight, and bi-level quantize with only the remaining weights.
                    # Do only first column on group
                
                weight_quant_i = quant_int(
                    weight[:, column_index].unsqueeze(1), quantizer.scale, quantizer.zero, quantizer.maxq
                )
                weight_i_quantized = dequant_int(weight_quant_i, quantizer.scale, quantizer.zero).reshape_as(
                    weight[:, column_index]
                )

                delta_weight_i = weight[:, column_index] - weight_i_quantized  # [out_dim]
                quantization_errors[:, column_index] = (
                    delta_weight_i / H_inv_cho[column_index, column_index]
                )  # [out_dim]

                if unstructured_outlier_threshold != float("inf"):
                    unstructured_outlier_mask[:, column_index] = (
                        quantization_errors[:, column_index].square() > unstructured_outlier_threshold
                    )
                    # re-quantize without outliers
                    is_outlier = unstructured_outlier_mask[:, column_index].float()

                    weight_quant_i = quant_int(
                        (weight[:,column_index] * (1 - is_outlier)).unsqueeze(1),
                        quantizer.scale,
                        quantizer.zero,
                        quantizer.maxq,
                    )
                    weight_i_quantized_wo_outliers = dequant_int(
                        weight_quant_i, quantizer.scale, quantizer.zero
                    ).reshape_as(weight[:, column_index])
                    weight_i_quantized = (
                        weight_i_quantized_wo_outliers * (1 - is_outlier) + weight[:, column_index] * is_outlier
                    )  # [out_dim]
                
                if save_quantization:
                    save_quant_dict["quant_weights"].append(weight_quant_i.to(torch.int8))

                weight[:, column_index] = weight_i_quantized
                weight[:, column_index + 1 : block_end].addr_(
                    quantization_errors[:, column_index],
                    H_inv_cho[column_index, column_index + 1 : block_end],
                    alpha=-1,
                )
            
            weight[:, block_end:].addmm_(
                quantization_errors[:, block_start:block_end],
                H_inv_cho[block_start:block_end, block_end:],
                alpha=-1,
            )

        if permutation_order != "identity":
            invperm = torch.argsort(perm)
            weight = weight[:, invperm]
        
        if save_quantization:
            save_quant_dict["perm"] = perm.to(torch.int32)
            save_quant_dict["keep_last_columns"] = 0
            save_quant_dict["blocksize"] = 128
            save_quant_dict["weight_shape"] = weight.shape
            save_quant_dict["groupsize"] = groupsize if groupsize else weight.shape[1]
            save_quant_dict["quant_weights"] = torch.cat(save_quant_dict["quant_weights"], dim=1)
            save_quant_dict["outliers_matrix"] = save_quant_dict["outliers_matrix"].to_sparse()
        
        return QuantizationResult(
            weight=weight,
            perm=perm,
            quantization_errors=quantization_errors,
            unstructured_outlier_threshold=unstructured_outlier_threshold,
            unstructured_outlier_mask=unstructured_outlier_mask,
            save_quant_dict=save_quant_dict,
        )