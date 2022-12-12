from argparse import ArgumentParser

import mmengine


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('project', help='Project name')
    args = parser.parse_args()
    return args


def fix_categories(path):
    data = mmengine.load(path)
    data['categories'] = data['categories'][1:]
    return data


if __name__ == '__main__':
    args = parse_args()
    mmengine.dump(
        fix_categories(f'data/{args.project}/train/_annotations.coco.json'),
        f'data/{args.project}/train/_annotations.coco.json')
    mmengine.dump(
        fix_categories(f'data/{args.project}/valid/_annotations.coco.json'),
        f'data/{args.project}/valid/_annotations.coco.json')
    mmengine.dump(
        fix_categories(f'data/{args.project}/test/_annotations.coco.json'),
        f'data/{args.project}/test/_annotations.coco.json')
