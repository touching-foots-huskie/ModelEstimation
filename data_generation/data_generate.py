import os
import sys
root_path = os.path.dirname(os.path.realpath(__file__))
# root_path = '/home/harvey/Projects/ModelEstimation/data_generation'
sys.path.append(root_path)

import random
import json
import bpy
import yaml

import blender_object
import label_compose
import file_batch_process


def main():
    # Path Configurations
    data_path = os.path.join(root_path, '../data')

    # Set configuration
    config_file_path = os.path.join(root_path, 'config/data_generate.yaml')
    with open(config_file_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    view_num = config['camera']['num_poses']
    iter_num = config['iter_num']
    data_num = iter_num*view_num
    class_num = len(config['obj_model_path']) + 1  # Include environment object

    # Clean previous data
    file_batch_process.file_batch_restart(data_path)

    for iter_ in range(iter_num):
        blender_object.remove_all()

        # Build environment
        env_object = blender_object.ModelObject(
            config['env_model_name'],
            os.path.join(root_path, config['env_model_path']), 
            1)
        env_object.set_passive_rigid()

        # Build objects
        range_x = config['params']['range_x']
        range_y = config['params']['range_y']
        range_z = config['params']['range_z']

        objects = list()
        class_offset = 2  # Offset for environment and start from 1
        for class_id, (object_name, object_path) in \
                enumerate(zip(config['obj_model_name'], config['obj_model_path'])):
            object_instance = blender_object.ModelObject(
                object_name, os.path.join(root_path, object_path), class_id + class_offset)
            object_instance.set_active_rigid()
            object_instance.random_pose_rigid(range_x, range_y, range_z)
            objects.append(object_instance)

        # Build light
        light_range_x = config['params']['light_position_range_x']
        light_range_y = config['params']['light_position_range_y']
        light_range_z = config['params']['light_position_range_z']
        
        light_object = blender_object.LightObject()
        light_object.random_pose(light_range_x, light_range_y, light_range_z)
        light_object.random_light()

        # Build camera
        cam_intrinsic = config['camera']['camera_intrinsics']
        cam_extrinsic = config['camera']['camera_poses']
        
        camera_object = blender_object.CameraObject(cam_intrinsic, cam_extrinsic, view_num)
        
        # Render view
        for cam_view_ in range(view_num):
            camera_object.set_pose_by_view(cam_view_)
            data_paths = label_compose.compose_node_gen(data_path, list(range(1, 1+class_num)))
            frame_number = random.randint(1, 10)  # Render in different frame number
            for step_ in range(frame_number + 1):
                bpy.context.scene.frame_set(step_)
            bpy.ops.render.render(layer='View Layer', scene='Scene')
            
            # Log new data
            for data_path_ in data_paths:
                log_data = dict()
                with open(os.path.join(data_path_, 'log.json'), 'r+') as f:
                    log_data = json.load(f)
                    log_data['num'] += 1
                    log_data['new_add'] = os.path.join(data_path_,'Image{:0>4d}.png'.format(frame_number))      
                with open(os.path.join(data_path_, 'log.json'), 'w') as f:
                    json.dump(log_data, f)
            
            # Rename data in our way
            file_batch_process.file_batch_rename(data_path)
    
    # Generate mask
    file_batch_process.mask_img_generation(data_path, class_num, data_num)
    os.remove(os.path.join(data_path, 'image', 'log.json'))  # Clean log files


if __name__ == '__main__':
    main()