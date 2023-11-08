import torch
import diffusers
import diffusers.utils.torch_utils
from typing import Optional, Union, Tuple


def PNDMScheduler__get_prev_sample(self, sample: torch.FloatTensor, timestep, prev_timestep, model_output):
    # See formula (9) of PNDM paper https://arxiv.org/pdf/2202.09778.pdf
    # this function computes x_(t−δ) using the formula of (9)
    # Note that x_t needs to be added to both sides of the equation

    # Notation (<variable name> -> <name in paper>
    # alpha_prod_t -> α_t
    # alpha_prod_t_prev -> α_(t−δ)
    # beta_prod_t -> (1 - α_t)
    # beta_prod_t_prev -> (1 - α_(t−δ))
    # sample -> x_t
    # model_output -> e_θ(x_t, t)
    # prev_sample -> x_(t−δ)
    sample.__str__() # PNDM Sampling does not work without 'stringify'. (because it depends on PLMS)
    alpha_prod_t = self.alphas_cumprod[timestep]
    alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev

    if self.config.prediction_type == "v_prediction":
        model_output = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
    elif self.config.prediction_type != "epsilon":
        raise ValueError(
            f"prediction_type given as {self.config.prediction_type} must be one of `epsilon` or `v_prediction`"
        )

    # corresponds to (α_(t−δ) - α_t) divided by
    # denominator of x_t in formula (9) and plus 1
    # Note: (α_(t−δ) - α_t) / (sqrt(α_t) * (sqrt(α_(t−δ)) + sqr(α_t))) =
    # sqrt(α_(t−δ)) / sqrt(α_t))
    sample_coeff = (alpha_prod_t_prev / alpha_prod_t) ** (0.5)

    # corresponds to denominator of e_θ(x_t, t) in formula (9)
    model_output_denom_coeff = alpha_prod_t * beta_prod_t_prev ** (0.5) + (
        alpha_prod_t * beta_prod_t * alpha_prod_t_prev
    ) ** (0.5)

    # full formula (9)
    prev_sample = (
        sample_coeff * sample - (alpha_prod_t_prev - alpha_prod_t) * model_output / model_output_denom_coeff
    )

    return prev_sample


diffusers.PNDMScheduler._get_prev_sample = PNDMScheduler__get_prev_sample # pylint: disable=protected-access


def UniPCMultistepScheduler_multistep_uni_p_bh_update(
    self: diffusers.UniPCMultistepScheduler,
    model_output: torch.FloatTensor,
    prev_timestep: int,
    sample: torch.FloatTensor,
    order: int,
) -> torch.FloatTensor:
    """
    One step for the UniP (B(h) version). Alternatively, `self.solver_p` is used if is specified.

    Args:
        model_output (`torch.FloatTensor`):
            direct outputs from learned diffusion model at the current timestep.
        prev_timestep (`int`): previous discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            current instance of sample being created by diffusion process.
        order (`int`): the order of UniP at this step, also the p in UniPC-p.

    Returns:
        `torch.FloatTensor`: the sample tensor at the previous timestep.
    """
    timestep_list = self.timestep_list
    model_output_list = self.model_outputs

    s0, t = self.timestep_list[-1], prev_timestep
    m0 = model_output_list[-1]
    x = sample

    if self.solver_p:
        x_t = self.solver_p.step(model_output, s0, x).prev_sample
        return x_t

    sample.__str__() # UniPC Sampling does not work without 'stringify'.
    lambda_t, lambda_s0 = self.lambda_t[t], self.lambda_t[s0]
    alpha_t, alpha_s0 = self.alpha_t[t], self.alpha_t[s0]
    sigma_t, sigma_s0 = self.sigma_t[t], self.sigma_t[s0]

    h = lambda_t - lambda_s0
    device = sample.device

    rks = []
    D1s = []
    for i in range(1, order):
        si = timestep_list[-(i + 1)]
        mi = model_output_list[-(i + 1)]
        lambda_si = self.lambda_t[si]
        rk = (lambda_si - lambda_s0) / h
        rks.append(rk)
        D1s.append((mi - m0) / rk)

    rks.append(1.0)
    rks = torch.tensor(rks, device=device)

    R = []
    b = []

    hh = -h if self.predict_x0 else h
    h_phi_1 = torch.expm1(hh)  # h\phi_1(h) = e^h - 1
    h_phi_k = h_phi_1 / hh - 1

    factorial_i = 1

    if self.config.solver_type == "bh1":
        B_h = hh
    elif self.config.solver_type == "bh2":
        B_h = torch.expm1(hh)
    else:
        raise NotImplementedError()

    for i in range(1, order + 1):
        R.append(torch.pow(rks, i - 1))
        b.append(h_phi_k * factorial_i / B_h)
        factorial_i *= i + 1
        h_phi_k = h_phi_k / hh - 1 / factorial_i

    R = torch.stack(R)
    b = torch.tensor(b, device=device)

    if len(D1s) > 0:
        D1s = torch.stack(D1s, dim=1)  # (B, K)
        # for order 2, we use a simplified version
        if order == 2:
            rhos_p = torch.tensor([0.5], dtype=x.dtype, device=device)
        else:
            rhos_p = torch.linalg.solve(R[:-1, :-1], b[:-1])
    else:
        D1s = None

    if self.predict_x0:
        x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
        if D1s is not None:
            pred_res = torch.einsum("k,bkchw->bchw", rhos_p, D1s)
        else:
            pred_res = 0
        x_t = x_t_ - alpha_t * B_h * pred_res
    else:
        x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
        if D1s is not None:
            pred_res = torch.einsum("k,bkchw->bchw", rhos_p, D1s)
        else:
            pred_res = 0
        x_t = x_t_ - sigma_t * B_h * pred_res

    x_t = x_t.to(x.dtype)
    return x_t


diffusers.UniPCMultistepScheduler.multistep_uni_p_bh_update = UniPCMultistepScheduler_multistep_uni_p_bh_update


def LCMScheduler_step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[diffusers.schedulers.scheduling_lcm.LCMSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_lcm.LCMSchedulerOutput`] or `tuple`.
        Returns:
            [`~schedulers.scheduling_utils.LCMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_lcm.LCMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.
        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # 1. get previous step value
        prev_step_index = self.step_index + 1
        if prev_step_index < len(self.timesteps):
            prev_timestep = self.timesteps[prev_step_index]
        else:
            prev_timestep = timestep

        # 2. compute alphas, betas
        sample.__str__()
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # 3. Get scalings for boundary conditions
        c_skip, c_out = self.get_scalings_for_boundary_condition_discrete(timestep)

        # 4. Compute the predicted original sample x_0 based on the model parameterization
        if self.config.prediction_type == "epsilon":  # noise-prediction
            predicted_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
        elif self.config.prediction_type == "sample":  # x-prediction
            predicted_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":  # v-prediction
            predicted_original_sample = alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction` for `LCMScheduler`."
            )

        # 5. Clip or threshold "predicted x_0"
        if self.config.thresholding:
            predicted_original_sample = self._threshold_sample(predicted_original_sample)
        elif self.config.clip_sample:
            predicted_original_sample = predicted_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 6. Denoise model output using boundary conditions
        denoised = c_out * predicted_original_sample + c_skip * sample

        # 7. Sample and inject noise z ~ N(0, I) for MultiStep Inference
        # Noise is not used for one-step sampling.
        if len(self.timesteps) > 1:
            noise = diffusers.utils.torch_utils.randn_tensor(model_output.shape, generator=generator, device=model_output.device)
            prev_sample = alpha_prod_t_prev.sqrt() * denoised + beta_prod_t_prev.sqrt() * noise
        else:
            prev_sample = denoised

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample, denoised)

        return diffusers.schedulers.scheduling_lcm.LCMSchedulerOutput(prev_sample=prev_sample, denoised=denoised)


diffusers.LCMScheduler.step = LCMScheduler_step
