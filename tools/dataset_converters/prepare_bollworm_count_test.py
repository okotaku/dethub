import cv2
import mmengine
import pandas as pd
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


def df2coco(imgs):
    img_infos = []
    img_id = 0
    for img_path in tqdm(imgs):
        img = cv2.imread(f'data/bollworm_count/images/{img_path}')
        img_info = dict(
            id=img_id,
            width=img.shape[1],
            height=img.shape[0],
            file_name=img_path,
        )
        img_infos.append(img_info)
        img_id += 1

    coco = init_coco()
    coco['images'] = img_infos
    return coco


def main():
    df = pd.read_csv('data/bollworm_count/Test.csv')
    img_ids = df.image_id_worm.unique()

    train_coco = df2coco(img_ids)
    mmengine.dump(train_coco, 'data/bollworm_count/dtest.json')


if __name__ == '__main__':
    main()
