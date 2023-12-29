from collections import defaultdict
import torch
import intel_extension_for_pytorch as ipex # pylint: disable=import-error, unused-import
import intel_extension_for_pytorch._C as core # pylint: disable=import-error, unused-import

# pylint: disable=protected-access, missing-function-docstring, line-too-long

device_supports_fp64 = torch.xpu.has_fp64_dtype()
OptState = ipex.cpu.autocast._grad_scaler.OptState
_MultiDeviceReplicator = ipex.cpu.autocast._grad_scaler._MultiDeviceReplicator
_refresh_per_optimizer_state = ipex.cpu.autocast._grad_scaler._refresh_per_optimizer_state

def _unscale_grads_(self, optimizer, inv_scale, found_inf, allow_fp16): # pylint: disable=unused-argument
    per_device_inv_scale = _MultiDeviceReplicator(inv_scale)
    per_device_found_inf = _MultiDeviceReplicator(found_inf)

    # To set up _amp_foreach_non_finite_check_and_unscale_, split grads by device and dtype.
    # There could be hundreds of grads, so we'd like to iterate through them just once.
    # However, we don't know their devices or dtypes in advance.

    # https://stackoverflow.com/questions/5029934/defaultdict-of-defaultdict
    # Google says mypy struggles with defaultdicts type annotations.
    per_device_and_dtype_grads = defaultdict(lambda: defaultdict(list))  # type: ignore[var-annotated]
    # sync grad to master weight
    if hasattr(optimizer, "sync_grad"):
        optimizer.sync_grad()
    with torch.no_grad():
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                if (not allow_fp16) and param.grad.dtype == torch.float16:
                    raise ValueError("Attempting to unscale FP16 gradients.")
                if param.grad.is_sparse:
                    # is_coalesced() == False means the sparse grad has values with duplicate indices.
                    # coalesce() deduplicates indices and adds all values that have the same index.
                    # For scaled fp16 values, there's a good chance coalescing will cause overflow,
                    # so we should check the coalesced _values().
                    if param.grad.dtype is torch.float16:
                        param.grad = param.grad.coalesce()
                    to_unscale = param.grad._values()
                else:
                    to_unscale = param.grad

                # -: is there a way to split by device and dtype without appending in the inner loop?
                to_unscale = to_unscale.to("cpu")
                per_device_and_dtype_grads[to_unscale.device][
                    to_unscale.dtype
                ].append(to_unscale)

        for _, per_dtype_grads in per_device_and_dtype_grads.items():
            for grads in per_dtype_grads.values():
                core._amp_foreach_non_finite_check_and_unscale_(
                    grads,
                    per_device_found_inf.get("cpu"),
                    per_device_inv_scale.get("cpu"),
                )

    return per_device_found_inf._per_device_tensors

def unscale_(self, optimizer):
    """
    Divides ("unscales") the optimizer's gradient tensors by the scale factor.
    :meth:`unscale_` is optional, serving cases where you need to
    :ref:`modify or inspect gradients<working-with-unscaled-gradients>`
    between the backward pass(es) and :meth:`step`.
    If :meth:`unscale_` is not called explicitly,  gradients will be unscaled  automatically during :meth:`step`.
    Simple example, using :meth:`unscale_` to enable clipping of unscaled gradients::
        ...
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()
    Args:
        optimizer (torch.optim.Optimizer):  Optimizer that owns the gradients to be unscaled.
    .. warning::
        :meth:`unscale_` should only be called once per optimizer per :meth:`step` call,
        and only after all gradients for that optimizer's assigned parameters have been accumulated.
        Calling :meth:`unscale_` twice for a given optimizer between each :meth:`step` triggers a RuntimeError.
    .. warning::
        :meth:`unscale_` may unscale sparse gradients out of place, replacing the ``.grad`` attribute.
    """
    if not self._enabled:
        return

    self._check_scale_growth_tracker("unscale_")

    optimizer_state = self._per_optimizer_states[id(optimizer)]

    if optimizer_state["stage"] is OptState.UNSCALED: # pylint: disable=no-else-raise
        raise RuntimeError(
            "unscale_() has already been called on this optimizer since the last update()."
        )
    elif optimizer_state["stage"] is OptState.STEPPED:
        raise RuntimeError("unscale_() is being called after step().")

    # FP32 division can be imprecise for certain compile options, so we carry out the reciprocal in FP64.
    assert self._scale is not None
    if device_supports_fp64:
        inv_scale = self._scale.double().reciprocal().float()
    else:
        inv_scale = self._scale.to("cpu").double().reciprocal().float().to(self._scale.device)
    found_inf = torch.full(
        (1,), 0.0, dtype=torch.float32, device=self._scale.device
    )

    optimizer_state["found_inf_per_device"] = self._unscale_grads_(
        optimizer, inv_scale, found_inf, False
    )
    optimizer_state["stage"] = OptState.UNSCALED

def update(self, new_scale=None):
    """
    Updates the scale factor.
    If any optimizer steps were skipped the scale is multiplied by ``backoff_factor``
    to reduce it. If ``growth_interval`` unskipped iterations occurred consecutively,
    the scale is multiplied by ``growth_factor`` to increase it.
    Passing ``new_scale`` sets the new scale value manually. (``new_scale`` is not
    used directly, it's used to fill GradScaler's internal scale tensor. So if
    ``new_scale`` was a tensor, later in-place changes to that tensor will not further
    affect the scale GradScaler uses internally.)
    Args:
        new_scale (float or :class:`torch.FloatTensor`, optional, default=None):  New scale factor.
    .. warning::
        :meth:`update` should only be called at the end of the iteration, after ``scaler.step(optimizer)`` has
        been invoked for all optimizers used this iteration.
    """
    if not self._enabled:
        return

    _scale, _growth_tracker = self._check_scale_growth_tracker("update")

    if new_scale is not None:
        # Accept a new user-defined scale.
        if isinstance(new_scale, float):
            self._scale.fill_(new_scale)  # type: ignore[union-attr]
        else:
            reason = "new_scale should be a float or a 1-element torch.FloatTensor with requires_grad=False."
            assert isinstance(new_scale, torch.FloatTensor), reason  # type: ignore[attr-defined]
            assert new_scale.numel() == 1, reason
            assert new_scale.requires_grad is False, reason
            self._scale.copy_(new_scale)  # type: ignore[union-attr]
    else:
        # Consume shared inf/nan data collected from optimizers to update the scale.
        # If all found_inf tensors are on the same device as self._scale, this operation is asynchronous.
        found_infs = [
            found_inf.to(device="cpu", non_blocking=True)
            for state in self._per_optimizer_states.values()
            for found_inf in state["found_inf_per_device"].values()
        ]

        assert len(found_infs) > 0, "No inf checks were recorded prior to update."

        found_inf_combined = found_infs[0]
        if len(found_infs) > 1:
            for i in range(1, len(found_infs)):
                found_inf_combined += found_infs[i]

        to_device = _scale.device
        _scale = _scale.to("cpu")
        _growth_tracker = _growth_tracker.to("cpu")

        core._amp_update_scale_(
            _scale,
            _growth_tracker,
            found_inf_combined,
            self._growth_factor,
            self._backoff_factor,
            self._growth_interval,
        )

        _scale = _scale.to(to_device)
        _growth_tracker = _growth_tracker.to(to_device)
    # To prepare for next iteration, clear the data collected from optimizers this iteration.
    self._per_optimizer_states = defaultdict(_refresh_per_optimizer_state)

def gradscaler_init():
    torch.xpu.amp.GradScaler = ipex.cpu.autocast._grad_scaler.GradScaler
    torch.xpu.amp.GradScaler._unscale_grads_ = _unscale_grads_
    torch.xpu.amp.GradScaler.unscale_ = unscale_
    torch.xpu.amp.GradScaler.update = update
    return torch.xpu.amp.GradScaler
