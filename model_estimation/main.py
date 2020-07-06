# Append the directory for torchvision
import sys
sys.path.append('/home/harvey/Softwares/vision/references/detection')

import time
import os
import yaml

# Visualization part
import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

from engine import train_one_epoch, evaluate
import utils
import transforms as T
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads

import blender_dataset
import visualize


'''
Initial Weight
'''
def init_weights(m):
    torch.nn.init.xavier_uniform(m.weight)


'''
Create Model
'''
def get_model_instance_segmentation(class_num):
    # Load pre-trained segmentation model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # Replace the box prediction head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, class_num)

    # Replace the mask prediction head
    # mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       class_num)
    
    # mask head
    out_channels = model.backbone.out_channels
    mask_layers = (256, 256, 256, 256)
    mask_dilation = 1
    model.roi_heads.mask_head = MaskRCNNHeads(out_channels,
                                              mask_layers, 
                                              mask_dilation)
    
    # change NMS threshold and detection number
    model.roi_heads.nms_thresh = 0.1
    model.roi_heads.detections_per_img = 10
    return model


'''
Data Transformation
'''
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


'''
Evaluation method
'''
@torch.no_grad()
def visual_evaluate(model, dataset, device, writer, class_names, epoch):
    cpu_device = torch.device('cpu')
    model.eval()
    num_test = min(len(dataset), 20)
    num_data = len(dataset)
    for idx in range(num_test):
        image, target = dataset[num_data-idx-1]  # target is the ground truth
        image = image.to(device)
        # get outputs
        output = model([image])
        output = {k: v.to(cpu_device) for k, v in output[0].items()}
        
        image = target['raw_image'].numpy()
        boxes = output['boxes'].numpy()
        masks = output['masks'].numpy()
        masks = np.squeeze(masks)
        labels = output['labels'].numpy()
        scores = output['scores'].numpy()

        gt_boxes = target['boxes'].numpy()
        gt_masks = target['masks'].numpy()
        gt_labels = target['labels'].numpy()
        
        # box check
        # Add training result visualization
        visualize.display_instances_tensorboard(
            image, boxes, masks, labels, class_names, writer,
            scores=None,
            show_mask=True, 
            show_bbox=True,
            title='Val:{}|epoch: {}'.format(idx, epoch))
        
        # Add Ground truth comparison visualization
        visualize.display_instances_tensorboard(
            image, gt_boxes, gt_masks, gt_labels, class_names, writer,
            scores=None,
            show_mask=True, 
            show_bbox=True,
            title='GT:{}|epoch: {}'.format(idx, epoch))


'''
Main function 
'''
def main():
    # Path configuration
    root_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(root_path, '../data')

    # Configuration Load
    config_file_path = os.path.join(root_path, 'config/model_estimation.yaml')
    with open(config_file_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    epoch_num = config['epoch_num']
    class_num = config['class_num'] + 1  # object class plus background

    # Train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Use our dataset and defined transformations
    dataset = blender_dataset.BlenderDataset(data_path, get_transform(train=True))
    dataset_test = blender_dataset.BlenderDataset(data_path, get_transform(train=False))
    dataset_vision = blender_dataset.BlenderDataset(data_path, get_transform(train=False))
    
    # Split 20 samples for test
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-20])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-20:])

    # Define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=2, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(class_num)
    # load model
    save_path = os.path.join(root_path, '..', config['model_path'], 'params.pkl')
    if config['parameter_load']:
        model.load_state_dict(torch.load(save_path))
        print('Model Loaded!')

    model.to(device)
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]  # take out grad-parameters
    # # previous optimizer
    # optimizer = torch.optim.SGD(params, lr=0.005,
    #                             momentum=0.9, weight_decay=0.0005)  
    # use Adam instead
    optimizer = torch.optim.Adam(params, lr=0.005, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    # clearn previous log add writter
    log_path = os.path.join(root_path, '..', config['log_path'], time.asctime())
    writer = SummaryWriter(log_path)
    # class names initialize
    class_names = ['background',
                   'table',
                   'up_glucose_bottle', 
                   'i_am_a_bunny_book',
                   'expo_dry_erase_board_eraser']
                   
    for epoch in range(epoch_num):
        # train for one epoch, printing every 10 iterations
        metric_logger = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        writer.add_scalar('Loss', metric_logger.loss.value)
        writer.add_scalar('Mask Loss', metric_logger.loss_mask.value)
        writer.add_scalar('Box Loss', metric_logger.loss_box_reg.value)
        writer.add_scalar('Class Loss', metric_logger.loss_classifier.value)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
    
    # make a evaluation after all
    visual_evaluate(model, dataset_vision, device, writer, class_names, 0)
    writer.close()
    # save the paramters
    if config['parameter_save']:
        torch.save(model.state_dict(), save_path)
        print('Model Saved!')


if __name__ == '__main__':
    main()