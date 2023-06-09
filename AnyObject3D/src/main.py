

import os 
import math
import cv2 
import numpy as np
import torch 
from PIL import Image
import argparse

from segment_anything import sam_model_registry, SamPredictor
from lavis.models import load_model_and_preprocess

from mono_rec import (gen_pc_from_image, gen_3d)


def rec_from_image(imgname, point, keyword, random_seed):
    # input params
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = cv2.imread(f'images/{imgname}.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sam_checkpoint = "./checkpoint/sam_vit_h_4b8939.pth"
    sam = sam_model_registry['vit_h'](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    predictor.set_image(image)

    input_point = np.array([point])
    # (1020, 430, 1950, 900)
    # prompt_boxes = np.array([550, 820, 1060, 1450])
    input_label = np.array([1])
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        # box = prompt_boxes,
        multimask_output=True,
    )

    mask = masks[np.argmax(scores)][:, :, None].astype(np.uint8) * 255
    res = np.where(mask > 0)
    y1, y2 = np.min(res[0]), np.max(res[0])
    x1, x2 = np.min(res[1]), np.max(res[1])
    cx, cy = (x1 + x2) / 2., (y1 + y2) / 2.
    w, h = x2 - x1, y2 - y1
    crop_sz = 1.8 * math.sqrt(w * h) # context_ratio
    x1, x2 = cx - crop_sz / 2., cx + crop_sz / 2.
    y1, y2 = cy - crop_sz / 2., cy + crop_sz / 2.
    x1, y1, x2, y2 = list(map(lambda x: int(x), [x1, y1, x2, y2]))

    # affine transform to get the object-centric image
    bbox = [x1, y1, x2, y2]
    output_sz = 512
    a = (output_sz-1) / (bbox[2]-bbox[0])
    b = (output_sz-1) / (bbox[3]-bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float32)
    img_patch = cv2.warpAffine(image, mapping, (output_sz, output_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    mask_patch = cv2.warpAffine(mask, mapping, (output_sz, output_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    img_patch = cv2.cvtColor(img_patch, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'./images/{imgname}_patch.jpg', img_patch)
    cv2.imwrite(f'./images/{imgname}_mask.png', mask_patch)
    img_patch[~(mask_patch > 0)] = [255, 255, 255]
    flag = cv2.imwrite(f'./images/{imgname}_masked.jpg', img_patch)

    # blip caption on image patch 
    raw_image = Image.open(f"./images/{imgname}_masked.jpg").convert("RGB")

    model, vis_processors, _ = load_model_and_preprocess(
        name="blip_caption", model_type="large_coco", is_eval=True, device=device
    )
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    output = model.generate({"image": image}, use_nucleus_sampling=True, num_captions=3)
    text_prompt = ''
    for prompt in output:
        if keyword in prompt:
            text_prompt = prompt 
            break
    text_prompt = keyword + ', ' + output[0] if len(text_prompt) == 0 else text_prompt
    print(keyword, ', ', text_prompt)
    images, points = gen_pc_from_image(imgname, text_prompt, keyword, random_seed)
    gen_3d(images, points, text_prompt, keyword, random_seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default=None, help="image name")
    parser.add_argument('--keyword', default=None, help="keyword")
    parser.add_argument('--seed', default=0, help="random seed")
    parser.add_argument(
        "--point_coords", type=float, nargs='+', required=True,
        help="The coordinate of the point prompt, [coord_W coord_H].",
    )
    opt = parser.parse_args()
    if opt.keyword is None:
        opt.keyword = opt.image
    import pdb 
    pdb.set_trace()
    rec_from_image(opt.image, opt.point_coords, opt.keyword, opt.seed)


