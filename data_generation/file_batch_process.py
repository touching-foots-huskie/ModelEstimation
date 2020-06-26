import os
import json
from PIL import Image
import numpy as np
import imageio
import shutil

def file_batch_rename(dir_path):
    sub_dirs = next(os.walk(dir_path))[1]
    for sub_dir in sub_dirs:
        sub_dir_full = os.path.join(dir_path, sub_dir)
        for file in os.listdir(sub_dir_full):
            if file.endswith('.json'):
                # check json file and find the newly added file
                log_data = dict()
                with open(os.path.join(sub_dir_full, file), 'r') as f:
                    log_data = json.load(f)
                
                with open(os.path.join(sub_dir_full, file), 'w') as f:
                    new_add_file = log_data['new_add']
                    img_num = log_data['num']
                    if new_add_file and os.path.isfile(os.path.join(sub_dir_full, new_add_file)):
                        os.rename(os.path.join(sub_dir_full, new_add_file),
                                  os.path.join(sub_dir_full, 'Img_{}.png'.format(img_num)))
                        log_data['new_add'] = None
                        json.dump(log_data, f)


def file_batch_restart(dir_path):
    # remove previous data
    mask_dir_path = os.path.join(dir_path, 'mask')
    if os.path.isdir(mask_dir_path):
        shutil.rmtree(mask_dir_path)

    sub_dirs = next(os.walk(dir_path))[1]
    for sub_dir in sub_dirs:
        sub_dir_full = os.path.join(dir_path, sub_dir)
        for file in os.listdir(sub_dir_full):
            file_name = os.path.join(sub_dir_full, file)
            os.remove(file_name)

        with open(os.path.join(sub_dir_full, 'log.json'), 'w+') as f:
            log_data = dict()
            log_data['num'] = -1
            log_data['new_add'] = None
            json.dump(log_data, f)


def mask_img_generation(dir_path, instance_number, img_number):
    if not os.path.isfile(os.path.join(dir_path, 'mask')):
        os.mkdir(os.path.join(dir_path, 'mask'))

    for id in range(img_number):
        mask_list = []
        for instance_id in range(1, instance_number+1):
            mask_path = os.path.join(dir_path, 'class_{}'.format(instance_id), 'Img_{}.png'.format(id))
            
            mask = Image.open(mask_path)
            mask = np.array(mask)
            mask = mask[:, :, 0]
            
            mask_data = np.zeros(mask.shape, dtype=np.uint8)
            mask_data[mask==255] = instance_id
            mask_list.append(mask_data)

        # Combine each label, different index for each class on pixel
        masks = np.stack(mask_list, axis=0)
        mask_img = masks.sum(axis=0).astype(dtype=np.uint8)
        output_path = os.path.join(dir_path, 'mask', 'Img_{}.png'.format(id))
        imageio.imwrite(output_path, mask_img)


if __name__ == '__main__':
    # file_batch_restart('home/harvey/Projects/ModelEstimation/data')
    mask_img_generation('home/harvey/Projects/ModelEstimation/data', 2, 63)