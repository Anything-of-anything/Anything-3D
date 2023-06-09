import torch
import numpy as np
from einops import rearrange
# from torch import autocast
from torch.cuda.amp import autocast
from contextlib import nullcontext
from math import sqrt
from adapt import ScoreAdapter

from cldm.model import create_model, load_state_dict
from lora_diffusion.cli_lora_add import *
from lora_diffusion.to_ckpt_v2 import *
import warnings
from transformers import logging
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.set_verbosity_error()

device = torch.device("cuda")


def lora_convert(model_path, as_half):
    
    """
    Modified version of lora_duffusion.to_ckpt_v2.convert_to_ckpt
    """

    assert model_path is not None, "Must provide a model path!"

    unet_path = osp.join(model_path, "unet", "diffusion_pytorch_model.bin")
    vae_path = osp.join(model_path, "vae", "diffusion_pytorch_model.bin")
    text_enc_path = osp.join(model_path, "text_encoder", "pytorch_model.bin")

    # Convert the UNet model
    unet_state_dict = torch.load(unet_path, map_location="cpu")
    unet_state_dict = convert_unet_state_dict(unet_state_dict)
    unet_state_dict = {
        "model.diffusion_model." + k: v for k, v in unet_state_dict.items()
    }

    # Convert the VAE model
    vae_state_dict = torch.load(vae_path, map_location="cpu")
    vae_state_dict = convert_vae_state_dict(vae_state_dict)
    vae_state_dict = {"first_stage_model." + k: v for k, v in vae_state_dict.items()}

    # Convert the text encoder model
    text_enc_dict = torch.load(text_enc_path, map_location="cpu")
    text_enc_dict = convert_text_enc_state_dict(text_enc_dict)
    text_enc_dict = {
        "cond_stage_model.transformer." + k: v for k, v in text_enc_dict.items()
    }

    # Put together new checkpoint
    state_dict = {**unet_state_dict, **vae_state_dict, **text_enc_dict}
    if as_half:
        state_dict = {k: v.half() for k, v in state_dict.items()}
    
    return state_dict

def merge(path_1: str,
    path_2: str,
    alpha_1: float = 0.5,
    ):

    loaded_pipeline = StableDiffusionPipeline.from_pretrained(
        path_1,
    ).to("cpu")

    tok_dict = patch_pipe(loaded_pipeline, path_2, patch_ti=False)

    collapse_lora(loaded_pipeline.unet, alpha_1)
    collapse_lora(loaded_pipeline.text_encoder, alpha_1)

    monkeypatch_remove_lora(loaded_pipeline.unet)
    monkeypatch_remove_lora(loaded_pipeline.text_encoder)
    
    _tmp_output = path_2[:-22]+"merge.tmp"

    loaded_pipeline.save_pretrained(_tmp_output)
    state_dict = lora_convert(_tmp_output, as_half=True)
    # remove the tmp_output folder
    shutil.rmtree(_tmp_output)

    keys = sorted(tok_dict.keys())
    tok_catted = torch.stack([tok_dict[k] for k in keys])
    ret = {
        "string_to_token": {"*": torch.tensor(265)},
        "string_to_param": {"*": tok_catted},
        "name": "",
    }

    return state_dict, ret



def _sqrt(x):
    if isinstance(x, float):
        return sqrt(x)
    else:
        assert isinstance(x, torch.Tensor)
        return torch.sqrt(x)

def load_embedding(model,embedding):
    length=len(embedding['string_to_param']['*'])
    voc=[]
    for i in range(length):
        voc.append(f'<{str(i)}>')
    print(f"Added Token: {voc}")
    model.cond_stage_model.tokenizer._add_tokens(voc)

    x=torch.nn.Embedding(model.cond_stage_model.tokenizer.__len__(),768)
    
    for params in x.parameters():
        params.requires_grad=False

    x.weight[:-length]=model.cond_stage_model.transformer.text_model.embeddings.token_embedding.weight
    x.weight[-length:]=embedding['string_to_param']['*']
    model.cond_stage_model.transformer.text_model.embeddings.token_embedding=x
    
def load_3DFuse(control,dir,alpha):
    ######################LOADCONTROL###########################
    model = create_model(control['control_yaml']).cpu()
    model.load_state_dict(load_state_dict(control['control_weight'], location='cuda'))
    state_dict, l = merge("runwayml/stable-diffusion-v1-5",dir,alpha)
    
    #######################OVERRIDE#############################
    model.load_state_dict(state_dict,strict=False)
    
    #######################ADDEMBBEDDING########################
    load_embedding(model,l)
    ###############################################################
    return model

class StableDiffusion(ScoreAdapter):
    def __init__(self, variant, v2_highres, prompt, scale, precision, dir, alpha=1.0):
               
        model=load_3DFuse(self.checkpoint_root(),dir,alpha)
        self.model = model.cuda()
        
        H , W = (512, 512)

        ae_resolution_f = 8

        self._device = self.model._device

        self.prompt = prompt
        self.scale = scale
        self.precision = precision
        self.precision_scope = autocast if self.precision == "autocast" else nullcontext
        self._data_shape = (4, H // ae_resolution_f, W // ae_resolution_f)

        self.cond_func = self.model.get_learned_conditioning
        self.M = 1000
        noise_schedule = "linear"
        self.noise_schedule = noise_schedule
        self.us = self.linear_us(self.M)

    def data_shape(self):
        return self._data_shape

    @property
    def σ_max(self):
        return self.us[0]

    @property
    def σ_min(self):
        return self.us[-1]

    @torch.no_grad()
    def denoise(self, xs, σ,control, **model_kwargs):
        with self.precision_scope(True):
        # with self.precision_scope("cuda"):
            with self.model.ema_scope():
                N = xs.shape[0]
                c = model_kwargs.pop('c')
                uc = model_kwargs.pop('uc')
                conditional_conditioning = {"c_concat": [control], "c_crossattn": [c]}
                unconditional_conditioning = {"c_concat": [control], "c_crossattn": [uc]}

                cond_t, σ = self.time_cond_vec(N, σ)
                unscaled_xs = xs
                xs = xs / _sqrt(1 + σ**2)
                if uc is None or self.scale == 1.:
                    output = self.model.apply_model(xs, cond_t, c)
                else:
                    x_in = torch.cat([xs] * 2)
                    t_in = torch.cat([cond_t] * 2)
                    c_in = dict()
                    for k in conditional_conditioning:
                        if isinstance(conditional_conditioning[k], list):
                            c_in[k] = [torch.cat([
                                unconditional_conditioning[k][i],
                                conditional_conditioning[k][i]]) for i in range(len(conditional_conditioning[k]))]
                        else:
                            c_in[k] = torch.cat([
                                    unconditional_conditioning[k],
                                    conditional_conditioning[k]])
                    
                    e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                    output = e_t_uncond + self.scale * (e_t - e_t_uncond)

                if self.model.parameterization == "v":
                    output = self.model.predict_eps_from_z_and_v(xs, cond_t, output)
                else:

                    output = output

                Ds = unscaled_xs - σ * output
                return Ds

    def cond_info(self, batch_size):
        prompts = batch_size * [self.prompt]
        return self.prompts_emb(prompts)

    @torch.no_grad()
    def prompts_emb(self, prompts):
        assert isinstance(prompts, list)
        batch_size = len(prompts)
        # with self.precision_scope("cuda"):
        with self.precision_scope(True):
            with self.model.ema_scope():
                cond = {}
                c = self.cond_func(prompts)
                cond['c'] = c
                uc = None
                if self.scale != 1.0:
                    uc = self.cond_func(batch_size * [""])
                cond['uc'] = uc
                return cond

    def unet_is_cond(self):
        return True

    def use_cls_guidance(self):
        return False

    def snap_t_to_nearest_tick(self, t):
        j = np.abs(t - self.us).argmin()
        return self.us[j], j

    def time_cond_vec(self, N, σ):
        if isinstance(σ, float):
            σ, j = self.snap_t_to_nearest_tick(σ)  # σ might change due to snapping
            cond_t = (self.M - 1) - j
            cond_t = torch.tensor([cond_t] * N, device=self.device)
            return cond_t, σ
        else:
            assert isinstance(σ, torch.Tensor)
            σ = σ.reshape(-1).cpu().numpy()
            σs = []
            js = []
            for elem in σ:
                _σ, _j = self.snap_t_to_nearest_tick(elem)
                σs.append(_σ)
                js.append((self.M - 1) - _j)

            cond_t = torch.tensor(js, device=self.device)
            σs = torch.tensor(σs, device=self.device, dtype=torch.float32).reshape(-1, 1, 1, 1)
            return cond_t, σs

    @staticmethod
    def linear_us(M=1000):
        assert M == 1000
        β_start = 0.00085
        β_end = 0.0120
        βs = np.linspace(β_start**0.5, β_end**0.5, M, dtype=np.float64)**2
        αs = np.cumprod(1 - βs)
        us = np.sqrt((1 - αs) / αs)
        us = us[::-1]
        return us

    @torch.no_grad()
    def encode(self, xs):
        model = self.model
        # with self.precision_scope("cuda"):
        with self.precision_scope(True):
            with self.model.ema_scope():
                zs = model.get_first_stage_encoding(
                    model.encode_first_stage(xs)
                )
        return zs

    @torch.no_grad()
    def decode(self, xs):
        # with self.precision_scope("cuda"):
        with self.precision_scope(True):
            with self.model.ema_scope():
                xs = self.model.decode_first_stage(xs)
                return xs
