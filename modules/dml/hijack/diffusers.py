from torch import FloatTensor
import diffusers

def _get_prev_sample(self, sample: FloatTensor, timestep, prev_timestep, model_output):
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
    sample.__str__() # DML Solution: PNDM Sampling does not work without 'stringify'. (because it depends on PLMS)
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

diffusers.PNDMScheduler._get_prev_sample = _get_prev_sample
