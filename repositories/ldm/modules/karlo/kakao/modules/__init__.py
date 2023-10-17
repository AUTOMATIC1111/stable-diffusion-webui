# ------------------------------------------------------------------------------------
# Adapted from Guided-Diffusion repo (https://github.com/openai/guided-diffusion)
# ------------------------------------------------------------------------------------


from .diffusion import gaussian_diffusion as gd
from .diffusion.respace import (
    SpacedDiffusion,
    space_timesteps,
)


def create_gaussian_diffusion(
    steps,
    learn_sigma,
    sigma_small,
    noise_schedule,
    use_kl,
    predict_xstart,
    rescale_learned_sigmas,
    timestep_respacing,
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
    )
