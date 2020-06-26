import bpy
import os


def compose_node_gen(data_path, class_ids):
    '''
    input: 
    class_ids, a list of class id
    '''
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.use_nodes = True
    bpy.context.scene.view_layers['View Layer'].use_pass_object_index = True

    tree = bpy.context.scene.node_tree
    for node in tree.nodes:
            tree.nodes.remove(node)

    render_layer_node = tree.nodes.new(type='CompositorNodeRLayers')
    object_index_socket = render_layer_node.outputs['IndexOB']
    full_img_socket = render_layer_node.outputs['Image']
    
    # File output
    img_out_node = tree.nodes.new(type='CompositorNodeOutputFile')
    img_out_node.base_path = os.path.join(data_path, 'image')
    raw_img_socket = img_out_node.inputs[0]
    tree.links.new(raw_img_socket, full_img_socket)

    # ID mask
    data_paths = [os.path.join(data_path, 'image')]
    for class_id in class_ids:
        id_mask_node = tree.nodes.new(type='CompositorNodeIDMask')
        id_mask_node.index = class_id
        id_mask_node.use_antialiasing = True
        id_value_socket = id_mask_node.inputs['ID value']
        alpha_socket = id_mask_node.outputs['Alpha']
        tree.links.new(id_value_socket, object_index_socket)

        file_out_node = tree.nodes.new(type='CompositorNodeOutputFile')
        file_out_node.base_path = os.path.join(data_path, 'class_{}'.format(class_id))
        img_socket = file_out_node.inputs['Image']
        tree.links.new(img_socket, alpha_socket)

        data_paths.append(os.path.join(data_path, 'class_{}'.format(class_id))) 
    return data_paths


if __name__ == '__main__':
    class_ids = [1, 2, 3]
    compose_node_gen('home/harvey/Projects/ModelEstimation', class_ids)