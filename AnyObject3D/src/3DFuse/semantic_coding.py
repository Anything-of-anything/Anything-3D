from diffusers import UnCLIPPipeline, DiffusionPipeline
import torch
import os
from lora_diffusion.cli_lora_pti import train
from PIL import Image
import numpy as np

def semantic_karlo(prompt, output_dir, num_initial_image, bg_preprocess, seed):
    pipe = UnCLIPPipeline.from_pretrained("kakaobrain/karlo-v1-alpha", torch_dtype=torch.float16)
    pipe = pipe.to('cuda')
    view_prompt=["front view of ","overhead view of ","side view of ", "back view of "]
    
    if bg_preprocess:
        # Please refer to the code at https://github.com/Ir1d/image-background-remove-tool.
        import cv2
        from carvekit.api.high import HiInterface
        interface = HiInterface(object_type="object",
                        batch_size_seg=5,
                        batch_size_matting=1,
                        device='cuda' if torch.cuda.is_available() else 'cpu',
                        seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
                        matting_mask_size=2048,
                        trimap_prob_threshold=231,
                        trimap_dilation=30,
                        trimap_erosion_iters=5,
                        fp16=False)
        
        

    generator = torch.Generator(device='cuda').manual_seed(seed)
    
    for i in range(num_initial_image):
        t=", white background" if bg_preprocess else ", white background"
        if i==0:
            prompt_ = f"{view_prompt[i%4]}{prompt}{t}"
        else:
            prompt_ = f"{view_prompt[i%4]}{prompt}"

        image = pipe(prompt_, generator=generator).images[0]
        fn=f"instance{i}.png"
        os.makedirs(output_dir,exist_ok=True)
        
        if bg_preprocess:
            # motivated by NeuralLift-360 (removing bg), and Zero-1-to-3 (removing bg and object-centering)
            # NOTE: This option was added during the code orgranization process.
            # The results reported in the paper were obtained with [bg_preprocess: False] setting.
            img_without_background = interface([image])
            mask = np.array(img_without_background[0]) > 127
            image = np.array(image)
            image[~mask] = [255., 255., 255.]
            # x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
            # image = image[y:y+h, x:x+w, :]
            image = Image.fromarray(np.array(image))
            
        image.save(os.path.join(output_dir,fn))
        
        
def semantic_sd(prompt, output_dir, num_initial_image, bg_preprocess, seed):
    pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe = pipe.to('cuda')
    view_prompt=["front view of ","overhead view of ","side view of ", "back view of "]
    
    if bg_preprocess:
        # Please refer to the code at https://github.com/Ir1d/image-background-remove-tool.
        import cv2
        from carvekit.api.high import HiInterface
        interface = HiInterface(object_type="object",
                        batch_size_seg=5,
                        batch_size_matting=1,
                        device='cuda' if torch.cuda.is_available() else 'cpu',
                        seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
                        matting_mask_size=2048,
                        trimap_prob_threshold=231,
                        trimap_dilation=30,
                        trimap_erosion_iters=5,
                        fp16=False)

    generator = torch.Generator(device='cuda').manual_seed(seed)
    for i in range(num_initial_image):
        t=", white background" if bg_preprocess else ", white background"
        if i==0:
            prompt_ = f"{view_prompt[i%4]}{prompt}{t}"
        else:
            prompt_ = f"{view_prompt[i%4]}{prompt}"

        image = pipe(prompt_, generator=generator).images[0]
        fn=f"instance{i}.png"
        os.makedirs(output_dir,exist_ok=True)
        
        if bg_preprocess:
            # motivated by NeuralLift-360 (removing bg), and Zero-1-to-3 (removing bg and object-centering)
            # NOTE: This option was added during the code orgranization process.
            # The results reported in the paper were obtained with [bg_preprocess: False] setting.
            img_without_background = interface([image])
            mask = np.array(img_without_background[0]) > 127
            image = np.array(image)
            image[~mask] = [255., 255., 255.]
            # x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
            # image = image[y:y+h, x:x+w, :]
            image = Image.fromarray(np.array(image))
            
        image.save(os.path.join(output_dir,fn))

def semantic_coding(exp_dir,cfgs,sd,initial):
    ti_step=cfgs.pop('ti_step')
    pt_step=cfgs.pop('pt_step')
    # semantic_model=cfgs.pop('semantic_model')
    prompt=cfgs['sd']['prompt']
    
    instance_dir=os.path.join(exp_dir,'initial_image')
    weight_dir=os.path.join(exp_dir,'lora')
    if initial=="":
        initial=None
    
    train(pretrained_model_name_or_path='runwayml/stable-diffusion-v1-5',\
          instance_data_dir=instance_dir,output_dir=weight_dir,gradient_checkpointing=True,\
          scale_lr=True,lora_rank=1,cached_latents=False,save_steps=ti_step,\
          max_train_steps_ti=ti_step,max_train_steps_tuning=pt_step, use_template="object",\
          lr_warmup_steps=0, lr_warmup_steps_lora=100, placeholder_tokens="<0>", initializer_tokens=initial,\
          continue_inversion=True, continue_inversion_lr=1e-4,device="cuda:0",            
          )
    if initial is not None:
        sd.prompt=prompt.replace(initial,'<0>')
    else:
        sd.prompt="a <0>"
    