import mmengine


def fix_categories(path):
    data = mmengine.load(path)
    data['categories'] = [data['categories'][1]]
    return data


if __name__ == '__main__':
    mmengine.dump(
        fix_categories('data/le2i/train/_annotations.coco.json'),
        'data/le2i/train/_annotations.coco.json')
    mmengine.dump(
        fix_categories('data/le2i/valid/_annotations.coco.json'),
        'data/le2i/valid/_annotations.coco.json')
    mmengine.dump(
        fix_categories('data/le2i/test/_annotations.coco.json'),
        'data/le2i/test/_annotations.coco.json')
