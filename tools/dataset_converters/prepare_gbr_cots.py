import cv2
import mmcv
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

CATEGORIES = ['gbr']
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


def df2coco(df):
    img_infos = []
    ann_infos = []
    img_id = 0
    ann_id = 0
    for _, row in tqdm(df.iterrows()):
        img_path = f'video_{str(row["video_id"])}/' \
                   f'{str(row["video_frame"])}.jpg'
        img = cv2.imread(f'data/train_images/{img_path}')
        img_info = dict(
            id=img_id,
            width=img.shape[1],
            height=img.shape[0],
            file_name=img_path,
        )
        if len(row['annotations']) != 0:
            for ann in row['annotations']:
                b_width = ann['width']
                b_height = ann['height']

                # some boxes in COTS are outside the image height and width
                if (ann['x'] + b_width > 1280):
                    b_width = 1280 - ann['x']
                if (ann['y'] + b_height > 720):
                    b_height = 720 - ann['y']

                ann_info = dict(
                    id=ann_id,
                    image_id=img_id,
                    category_id=0,
                    iscrowd=0,
                    area=ann['width'] * ann['height'],
                    bbox=[ann['x'], ann['y'], b_width, b_height],
                    segmentation=[])
                ann_infos.append(ann_info)
                ann_id += 1
        img_infos.append(img_info)
        img_id += 1

    coco = init_coco()
    coco['images'] = img_infos
    coco['annotations'] = ann_infos
    return coco


def main():
    df = pd.read_csv('data/train.csv')
    df['annotations'] = df['annotations'].apply(eval)
    df['len_ann'] = df['annotations'].map(lambda x: len(x))
    df['has_label'] = (df['len_ann'] > 0) * 1

    df = df.sample(frac=1, random_state=0).reset_index(drop=True)
    kf = StratifiedGroupKFold(n_splits=5)
    splits = list(
        kf.split(df['has_label'], df['has_label'], df['sequence'].values))

    for fold, (train_inds, val_inds) in enumerate(splits):
        train_df = df.iloc[train_inds]
        train_coco = df2coco(train_df)
        mmcv.dump(train_coco, f'dtrain_g{fold}.json')

        val_df = df.iloc[val_inds]
        val_coco = df2coco(val_df)
        mmcv.dump(val_coco, f'dval_g{fold}.json')
        break


if __name__ == '__main__':
    main()
