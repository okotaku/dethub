import cv2
import mmengine
import numpy as np
import pandas as pd
from shapely.wkt import loads
from sklearn.model_selection import KFold
from tqdm import tqdm

CATEGORIES = ['pbw', 'abw']
CAT2IDX = {cat: idx for idx, cat in enumerate(CATEGORIES)}
ID_COL = 'image_id'


def init_coco():
    return {
        'info': {},
        'categories': [{
            'id': idx,
            'name': cat,
        } for cat, idx in CAT2IDX.items()]
    }


def df2coco(imgs, bboxes):
    img_infos = []
    ann_infos = []
    img_id = 0
    ann_id = 0
    for img_path in tqdm(imgs):
        #print(f'data/bollworm_count/images/{img_path}')
        img = cv2.imread(f'data/bollworm_count/images/{img_path}')
        img_info = dict(
            id=img_id,
            width=img.shape[1],
            height=img.shape[0],
            file_name=img_path,
        )
        bboxes_ = bboxes[bboxes.image_id == img_path]
        if img_path == 'id_7ea2112ddf0ffffe7e8ee272.jpg':
            print(img_path)
        if len(bboxes_) != 0:
            for geometry, category in zip(bboxes_.geometry.values, bboxes_.worm_type.values):
                bbox = loads(geometry).bounds

                x_min, y_min, x_max, y_max = bbox
                if x_max > img.shape[1]:
                    continue
                    #print("fix x", img_path, x_max, x_min, y_max, y_min, img.shape)
                    #x_max = img.shape[1]

                if y_max > img.shape[0]:
                    continue
                    #print("fix y", img_path, x_max, x_min, y_max, y_min, img.shape)
                    #y_max = img.shape[0]
                if x_max < x_min or y_max < y_min:
                    continue
                if x_min < 0 or y_min < 0:
                    continue

                ann_info = dict(
                    id=ann_id,
                    image_id=img_id,
                    category_id=CAT2IDX[category],
                    iscrowd=0,
                    area=(x_max - x_min) * (y_max - y_min),
                    bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                    segmentation=[])
                ann_infos.append(ann_info)
                ann_id += 1
        if img_path == 'id_7ea2112ddf0ffffe7e8ee272.jpg':
            print(bboxes_)
            break
        img_infos.append(img_info)
        img_id += 1

    print(ann_id, len(bboxes))
    #assert ann_id == len(bboxes)
    coco = init_coco()
    coco['images'] = img_infos
    coco['annotations'] = ann_infos
    return coco


def main():
    df = pd.read_csv('data/bollworm_count/Train.csv')
    bboxes = pd.read_csv('data/bollworm_count/images_bboxes.csv')
    bboxes = bboxes[~bboxes.worm_type.isna()]

    df = df.sample(frac=1, random_state=0).reset_index(drop=True)
    img_ids = df.image_id_worm.unique()
    kf = KFold(n_splits=5, shuffle=True, random_state=12345)
    splits = list(kf.split(img_ids))

    for fold, (train_inds, val_inds) in enumerate(splits):
        train_imgs = img_ids[train_inds]
        train_bboxes = bboxes[bboxes.image_id.isin(train_imgs)]
        train_coco = df2coco(train_imgs,train_bboxes)
        #mmengine.dump(train_coco, f'data/bollworm_count/dtrain_fold{fold}.json')

        val_imgs = img_ids[val_inds]
        val_bboxes = bboxes[bboxes.image_id.isin(val_imgs)]
        val_coco = df2coco(val_imgs, val_bboxes)
        #mmengine.dump(val_coco, f'data/bollworm_count/dval_fold{fold}.json')
        break


if __name__ == '__main__':
    main()
