from adapt_sd import StableDiffusion

from my.config import BaseConf


class SD(BaseConf):
    """Stable Diffusion"""
    variant:        str = "v1"
    v2_highres:     bool = False
    prompt:         str = "a photograph of an astronaut riding a horse"
    scale:          float = 3.0  # classifier free guidance scale
    precision:      str = 'autocast'
    dir:            str = './'
    alpha:          float = 0.0 # merge scale

    def make(self):
        args = self.dict()
        model = StableDiffusion(**args)
        return model