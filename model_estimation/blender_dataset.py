import os
import numpy as np
import torch
from PIL import Image


class BlenderDataset(object):
    def __init__(self, root_path, transforms):
        self.root_path = root_path
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root_path, "image"))))
        self.masks = list(sorted(os.listdir(os.path.join(root_path, "mask"))))

    def __getitem__(self, idx):
        # Load images ad masks
        img_path = os.path.join(self.root_path, "image", self.imgs[idx])
        mask_path = os.path.join(self.root_path, "mask", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = np.array(mask)
        
        # Instances are encoded as different colors
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]

        # Generate bounding box
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # Create tags for torch
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(obj_ids, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["raw_image"] = torch.from_numpy(np.array(img))

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)