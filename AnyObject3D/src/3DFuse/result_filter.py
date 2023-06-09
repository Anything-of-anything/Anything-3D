import os 
import glob
import os.path as osp
import shutil
import cv2 

def vis_mask(image, mask):
    heatmapshow = None
    heatmapshow = cv2.normalize(mask, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
    heatmapshow[mask==0] = [0, 0, 0]
    fused_img = cv2.addWeighted(heatmapshow, 0.2, image, 0.8, 0)
    return fused_img

if __name__ == '__main__':
    root_path = '/home/shenqiuhong/anything/3DFuse'
    result_path = osp.join(root_path, 'results')

    filtered_path = osp.join(result_path, 'filtered')
    if not os.path.exists(filtered_path):
        os.mkdir(filtered_path)


    for scene_path in os.listdir(result_path):
        scene_name = scene_path.split('_')[0]
        view_path = osp.join(result_path, scene_path, '3d', 'result_10000', 'img')
        if os.path.exists(view_path):
            print(scene_name)
            scene_dir = osp.join(filtered_path, scene_name)
            image = cv2.imread(osp.join(root_path, 'imgs', scene_name + '.jpg'))
            mask = cv2.imread(osp.join(root_path, 'img_masks', scene_name + '_mask.png'))
            masked_image = vis_mask(image, mask[:, :, 0])
            cv2.imwrite(osp.join(scene_dir, scene_name + '.jpg'), masked_image)
            if not os.path.exists(scene_dir):
                os.mkdir(scene_dir)
            view_imgs = glob.glob(osp.join(view_path, '*.png'))[::5][:8]
            for idx, imgpath in enumerate(view_imgs):
                shutil.copy(imgpath, osp.join(scene_dir, f'view_{idx}.png'))