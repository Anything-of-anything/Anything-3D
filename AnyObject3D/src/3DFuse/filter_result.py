import os 
import glob
import os.path as osp


if __name__ == '__main__':
    result_path = '/home/shenqiuhong/anything/3DFuse/results'
    # all_scenes = os.listdir(result_path)
    for scene_path in os.listdir(result_path):
        view_path = osp.join(result_path, scene_path, '3d', 'result_10000', 'img')
        if os.path.exists(view_path):
            print(scene_path)