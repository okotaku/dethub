import argparse

import mmengine

CATEGORIES = ('foreground', )
CAT2IDX = {cat: idx for idx, cat in enumerate(CATEGORIES)}


def parse_args():
    parser = argparse.ArgumentParser(description='Convert dataset to 1 class')

    parser.add_argument('input', help='input json file path')
    parser.add_argument('output', help='output json file path')

    return parser.parse_args()


def main():
    args = parse_args()
    data = mmengine.load(args.input)

    data['categories'] = [{
        'id': idx + 1,
        'name': cat,
    } for cat, idx in CAT2IDX.items()]

    for ann in data['annotations']:
        ann['category_id'] = 1

    mmengine.dump(data, args.output)


if __name__ == '__main__':
    main()
