import os
import bpy
import random
import math
from mathutils import Vector, Matrix, Quaternion
from functools import partial


class RawObject:
    def __init__(self):
        self.name = None

    def get_name(self):
        return self.name

    def get_object(self):
        return bpy.data.objects[self.name]

    def set_active(self):
        bpy.context.view_layer.objects.active = bpy.data.objects[self.name]
    
    def set_inactive(self):
        bpy.context.view_layer.objects.active = None

    def set_select(self, status):
        bpy.data.objects[self.name].select_set(status)

    def set_pose(self, pose):
        object_instance = self.get_object()
        object_instance.location = [pose[0], pose[1], pose[2]]
        object_instance.rotation_mode = 'QUATERNION'
        object_instance.rotation_quaternion = [pose[3], pose[4], pose[5], pose[6]]

    def random_pose(self, range_x, range_y, range_z):
        x = random.uniform(range_x[0], range_x[1])
        y = random.uniform(range_y[0], range_y[1])
        z = random.uniform(range_z[0], range_z[1])

        object_instance = self.get_object()
        object_instance.location = [x, y, z]
    
    def random_rotate(self):
        x = random.random()
        y = random.random()
        z = random.random()
        w = random.random()
        m = math.sqrt(x * x + y * y + z * z + w * w)
        x = x / m
        y = y / m
        z = z / m
        w = w / m
        object_instance = self.get_object()
        object_instance.rotation_mode = 'QUATERNION'
        object_instance.rotation_quaternion = [x, y, z, w]

    # def __del__(self):
    #     self.set_select(True)
    #     bpy.ops.object.delete(use_global=True)

# Sort function for key
def key_fn(x, name):
    tags = x.split('.')
    if tags[0] == name:
        if len(tags) == 1:
            return (1, 0)
        else:
            return (1, int(tags[1]))
    else:
        if len(tags) == 1:
            return (0, 0)
        else:
            return (0, int(tags[1]))

class ModelObject(RawObject):
    def __init__(self, object_model_name, object_model_path, class_id):
        RawObject.__init__(self)
        self.model_path = object_model_path
        bpy.ops.import_scene.obj(filepath=self.model_path)

        # Find the name and lasted added
        key_fn_name = partial(key_fn, name=object_model_name)
        sorted_name = sorted(bpy.data.objects.keys(), key=key_fn_name)
        self.name = sorted_name[-1]
        bpy.data.objects[self.name].pass_index = class_id
        self.set_select(False)
    
    def set_passive_rigid(self):
        self.set_active()
        bpy.ops.rigidbody.object_add(type='PASSIVE')
        self.set_inactive()

    def set_active_rigid(self):
        self.set_active()
        bpy.ops.rigidbody.object_add(type='ACTIVE')
        self.set_inactive()
    
    def set_pose_rigid(self, pose):
        # Use for rigid object
        object_instance = self.get_object()
        object_instance.rigid_body.kinematic = True
        self.set_pose(pose)
        object_instance.rigid_body.kinematic = False
     
    def random_pose_rigid(self, range_x, range_y, range_z):
        # Use for rigid object
        object_instance = self.get_object()
        # Set 0 before move
        bpy.context.scene.frame_set(0)
        object_instance.rigid_body.kinematic = True
        self.random_pose(range_x, range_y, range_z)
        self.random_rotate()
        object_instance.rigid_body.kinematic = False
    

class LightObject(RawObject):
    lightColors = [[255,197,142], [255,214,170], [255,241,224],
                   [255,250,244], [255,255,251], [255,255,255]]

    def __init__(self):
        RawObject.__init__(self)
        bpy.ops.object.light_add(type='POINT')
        self.name = bpy.data.lights[-1].name
        self.set_select(False)
    
    # def __del__(self):
    #     light_object = self.get_object()
    #     remove_light(light_object)
    
    def get_light(self):
        return bpy.data.lights[self.name]

    def random_light(self):
        light_instance = self.get_light()
        light_instance.energy = random.randint(0, 20)
        color_idx = random.randint(0, len(self.lightColors) - 1)
        light_instance.color = (self.lightColors[color_idx][0]/255,
                                self.lightColors[color_idx][1]/255,
                                self.lightColors[color_idx][2]/255)

    @staticmethod
    def clear_all_light():
        bpy.ops.object.select_by_type(type='LIGHT')
        bpy.ops.object.delete(use_global=False)


class CameraObject(RawObject):
    def __init__(self, cam_intrinsic, cam_extrinsic, num_views):
        RawObject.__init__(self)
        self.cam_intrinsic = cam_intrinsic
        self.cam_extrinsic = cam_extrinsic
        self.num_views = num_views
        sensor_width_in_mm = self.cam_intrinsic[1][1]*self.cam_intrinsic[0][2] \
                            / (self.cam_intrinsic[0][0]*self.cam_intrinsic[1][2])
        sensor_height_in_mm = 1
        resolution_x_in_px = self.cam_intrinsic[0][2]*2  # principal point assumed at the center
        resolution_y_in_px = self.cam_intrinsic[1][2]*2  # principal point assumed at the center

        s_u = resolution_x_in_px / sensor_width_in_mm
        s_v = resolution_y_in_px / sensor_height_in_mm
        f_in_mm = self.cam_intrinsic[0][0] / s_u

        # recover original resolution
        bpy.context.scene.render.resolution_x = resolution_x_in_px
        bpy.context.scene.render.resolution_y = resolution_y_in_px
        bpy.context.scene.render.resolution_percentage = 100

        # Create Camera
        bpy.ops.object.camera_add()
        self.name = bpy.data.cameras[-1].name
        self.set_pose([0, 0, 0, 1, 0, 0, 0])
        cam_data = self.get_camera()
        cam_data.type = 'PERSP'
        cam_data.lens = f_in_mm 
        cam_data.lens_unit = 'MILLIMETERS'
        cam_data.sensor_width  = sensor_width_in_mm
        cam_instance = self.get_object()
        cam_instance.data = cam_data
        bpy.context.scene.camera = cam_instance
        self.set_select(False)

    # def __del__(self):
    #     camera_object = self.get_object()
    #     remove_camera(camera_object)

    def get_camera(self):
        return bpy.data.cameras[self.name]

    def set_pose_by_view(self, view):
        bpy.data.objects['Camera'].location = (self.cam_extrinsic[view][0], 
                                               self.cam_extrinsic[view][1], 
                                               self.cam_extrinsic[view][2])
        bpy.data.objects['Camera'].rotation_mode = 'QUATERNION'
        bpy.data.objects['Camera'].rotation_quaternion = (self.cam_extrinsic[view][3], 
                                                          self.cam_extrinsic[view][4],
                                                          self.cam_extrinsic[view][5], 
                                                          self.cam_extrinsic[view][6])
    
    @staticmethod
    def clear_all_camera():
        bpy.ops.object.select_by_type(type='CAMERA')
        bpy.ops.object.delete(use_global=False)


# Ultilities
def deselect_all():
    for object in bpy.ops.objects:
        object.select_set(False)


def remove_camera(camera_object):
    camera_data = camera_object.data
    bpy.data.cameras.remove(camera_data)


def remove_light(light_object):
    light_data = light_object.data
    bpy.data.lights.remove(light_data)


def remove_all():   
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=True)

    for camera in bpy.data.cameras:
        bpy.data.cameras.remove(camera=camera)
    
    for light in bpy.data.lights:
        bpy.data.lights.remove(light=light)

    
