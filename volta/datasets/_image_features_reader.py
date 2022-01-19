
import copy
import lmdb  # install lmdb by "pip install lmdb"
import base64
import pickle
from typing import List
import sys

import numpy as np
import csv
import glob

FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id",
              "num_boxes", "boxes", "features",
              "cls_prob"]

import numpy as np
import os
import base64

def remove_suffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string

#currently top 10 highest items, as a test
def min_max_prob(path, item_to_check, n):
    """
    
    Input: directory of features,  item to check, n number of highest prediction boxes
    
    returns.... should be the same but with n number of highest predicted boxes and all of their equivalent dicts
    
    Loads each numpy file in path
    Each numpy file have dicts ['bbox', 'num_boxes', 'objects', 'image_height', 'image_width', 'cls_prob', 'features']
    
    
    """ 
    
    lowest = 100
    highest = 0
    listofvalues = []
    
    
    for filepath in glob.iglob(path):                     #for each image path in dir
        looplist = []
        out = np.load(filepath, allow_pickle=True)               #loads the image
        name = os.path.basename(filepath)
        name = str(name)
        name = remove_suffix(name, ".npy")   #name of file
        name = int(name)               #remove zeros
                        
        height = out.item()["image_height"]
        width = out.item()["image_width"]
        num_boxes = out.item()["num_boxes"]

        highestprobs = []
        for j in range(num_boxes):
            prob = max(out.item()['cls_prob'][j])
            prob_and_indices = (prob, j)
            highestprobs.append(prob_and_indices)
        
        sort_by_prob = sorted(highestprobs, reverse=True)[:n]
        indices_highestprobs = [y for x,y in sort_by_prob]
        

        
        each_image_info = []
        for i in indices_highestprobs:          #for each item in bounding box 

            img_id = filepath
            feat = out.item()["features"][i]               #features
            bbox = out.item()['bbox'][i]                   #bbox coordinates
            prob = out.item()['cls_prob'][i]          #highest probability object
            obj = out.item()['objects'][i]                 #object
            
            test_append = out.item()["bbox"][:n]        #just 10 boxes for test
            
            ##encoded same as in original function, "get_detections_from_im", https://github.com/e-bug/volta/blob/main/data/mscoco/extract_coco_image.py
            image_info =  {
        "img_id": name,                       #COCO_val2014_000000118343
        "img_h": height,
        "img_w": width,
        "objects_id": base64.b64encode(obj),  # int64
        "num_boxes": len(indices_highestprobs),
        "boxes": base64.b64encode(test_append),  # float32
        "features": base64.b64encode(feat),  # float32
        "cls_prob": base64.b64encode(prob),
    }
        
    
        listofvalues.append(image_info)
    
    #listofvalues = listofvalues[:51]  #remove 450 images
    return listofvalues


path_elephants = "/home/gushertni@GU.GU.SE/aicsproject/features/frcnn_feat_500cocoelephants/500_coco_elephants/*"

#elephant values used as input into ImageFeaturesH5Reader
elephantvalues = min_max_prob(path_elephants, "cls_prob", 10) #10 highest items as a test



class ImageFeaturesH5Reader(object):
    """
    A reader for H5 files containing pre-extracted image features. A typical
    H5 file is expected to have a column named "image_id", and another column
    named "features".
    Example of an H5 file:
    ```
    faster_rcnn_bottomup_features.h5
       |--- "image_id" [shape: (num_images, )]
       |--- "features" [shape: (num_images, num_proposals, feature_size)]
       +--- .attrs ("split", "train")
    ```
    # TODO (kd): Add support to read boxes, classes and scores.
    Parameters
    ----------
    features_h5path : str
        Path to an H5 file containing COCO train / val image features.
    in_memory : bool
        Whether to load the whole H5 file in memory. Beware, these files are
        sometimes tens of GBs in size. Set this to true if you have sufficient
        RAM - trade-off between speed and memory.
    """

    def __init__(self, features_path: str, config, in_memory: bool = False):
        self.features_path = features_path
        self._in_memory = in_memory
        
        
        

        # If not loaded in memory, then list of None.
        self.env = lmdb.open(
            self.features_path,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        with self.env.begin(write=False) as txn:
            self._image_ids = pickle.loads(txn.get("keys".encode()))

        self.features = [None] * len(self._image_ids)         #initializes empty arrays
        self.num_boxes = [None] * len(self._image_ids)
        self.boxes = [None] * len(self._image_ids)
        self.boxes_ori = [None] * len(self._image_ids)
        self.feature_size = 2048 #config.v_feature_size            
        self.num_locs = 5 #config.num_locs            
        self.add_global_imgfeat = config.add_global_imgfeat

    def __len__(self):
        return len(self._image_ids)

    def __getitem__(self, image_id):
        
        
        for item in elephantvalues:

            image_h = int(item["img_h"])
            image_w = int(item["img_w"])
            features = np.frombuffer(base64.b64decode(item["features"]), dtype=np.float32).reshape(-1, self.feature_size) 
            boxes = np.frombuffer(base64.b64decode(item['boxes']), dtype=np.float32).reshape(-1, 4)

            if features is not None:

                image_location = np.zeros((boxes.shape[0], self.num_locs), dtype=np.float32)
                image_location[:, :4] = boxes
                if self.num_locs == 5:
                    image_location[:, 4] = (
                            (image_location[:, 3] - image_location[:, 1])
                            * (image_location[:, 2] - image_location[:, 0])
                            / (float(image_w) * float(image_h))
                    )

                image_location_ori = copy.deepcopy(image_location)
                image_location[:, 0] = image_location[:, 0] / float(image_w)
                image_location[:, 1] = image_location[:, 1] / float(image_h)
                image_location[:, 2] = image_location[:, 2] / float(image_w)
                image_location[:, 3] = image_location[:, 3] / float(image_h)

                num_boxes = features.shape[0] 
                if self.add_global_imgfeat == "first":
                    g_feat = np.sum(features, axis=0) / num_boxes
                    num_boxes = num_boxes + 1
                    features = np.concatenate([np.expand_dims(g_feat, axis=0), features], axis=0)

                    g_location = [0, 0, 1, 1] + [1] * (self.num_locs - 4)         #
                    image_location = np.concatenate([np.expand_dims(g_location, axis=0), image_location], axis=0)

                    g_location_ori = np.array([0, 0, image_w, image_h] + [image_w * image_h] * (self.num_locs - 4))
                    image_location_ori = np.concatenate(
                        [np.expand_dims(g_location_ori, axis=0), image_location_ori], axis=0
                    )

                elif self.add_global_imgfeat == "last":
                    g_feat = np.sum(features, axis=0) / num_boxes
                    num_boxes = num_boxes + 1
                    features = np.concatenate([features, np.expand_dims(g_feat, axis=0)], axis=0)

                    g_location = [0, 0, 1, 1] + [1] * (self.num_locs - 4)
                    image_location = np.concatenate([image_location, np.expand_dims(g_location, axis=0)], axis=0)

                    g_location_ori = np.array([0, 0, image_w, image_h] + [image_w * image_h] * (self.num_locs - 4))
                    image_location_ori = np.concatenate(
                        [image_location_ori, np.expand_dims(g_location_ori, axis=0)], axis=0
                    )

            return features, num_boxes, image_location, image_location_ori          #indentation here?

    def keys(self) -> List[int]:
        return self._image_ids