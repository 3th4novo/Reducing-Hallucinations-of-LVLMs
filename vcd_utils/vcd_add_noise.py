import torch

# def add_diffusion_noise(image_tensor, noise_step):
#     num_steps = 1000  # Number of diffusion steps
#
#     # decide beta in each step
#     betas = torch.linspace(-6,6, num_steps)
#     betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
#
#     # decide alphas in each step
#     alphas = 1 - betas
#     alphas_prod = torch.cumprod(alphas, dim=0)
#     alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]],0) # p for previous
#     alphas_bar_sqrt = torch.sqrt(alphas_prod)
#     one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
#     one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)
#
#     def q_x(x_0,t,b=0.27):
#         noise = torch.randn_like(x_0)
#         alphas_t = alphas_bar_sqrt[t]
#         alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
#         return (alphas_t*b*x_0 + alphas_1_m_t*noise)
#
#     noise_delta = int(noise_step) # from 0-999
#     noisy_image = image_tensor.clone()
#     image_tensor_cd = q_x(noisy_image,noise_step)


def add_diffusion_noise(image_tensor, noise_step, s=0.008):
    time_steps = 1000  # Number of diffusion steps

    def cosine_beta_schedule(timesteps, s=0.008):
        """
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    # decide alphas in each step
    betas = cosine_beta_schedule(timesteps=1000)
    alphas = 1. - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)  # p for previous
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    def q_x(x_0, t, b=0.27):
        noise = torch.randn_like(x_0)
        alphas_t = alphas_bar_sqrt[t]
        alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
        return (alphas_t * b * x_0 + alphas_1_m_t * noise)

    noise_delta = int(noise_step)  # from 0-999
    noisy_image = image_tensor.clone()
    image_tensor_cd = q_x(noisy_image, noise_step)

    return image_tensor_cd

