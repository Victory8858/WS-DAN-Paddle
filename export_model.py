import os
import argparse

import paddle

from models.wsdan import WSDAN


def parse_args():
    parser = argparse.ArgumentParser(description='Fined-Grained Classification')
    parser.add_argument("--model", default="bird", type=str, help="bird, car, aircraft")
    parser.add_argument("--input_size", default=(448, 448), type=tuple)
    parser.add_argument('--save_dir', default='output', type=str, help='The directory for saving the exported model')

    return parser.parse_args()


def main(args):
    # model
    if args.model == 'bird':
        num_classes = 200
    elif args.models == 'car':
        num_classes = 196
    elif args.models == 'aircraft':
        num_classes = 100
    paddle.disable_static()
    model = WSDAN(num_classes=num_classes, num_attentions=32, net_name='inception_mixed_6e', pretrained=False)
    model_path = os.path.join('FGVC', args.model, args.model) + '_model.pdparams'
    print("model_path: ", model_path)
    net_state_dict = paddle.load(model_path)
    model.set_dict(net_state_dict)
    print('Loaded trained params of model successfully.')
    model.eval()

    model = paddle.jit.to_static(model, input_spec=[paddle.static.InputSpec(shape=[None, 3, 448, 448], dtype="float32")])
    # save inference model
    paddle.jit.save(model, os.path.join(args.save_dir, "inference"))
    print(
        f"inference model  have been saved into {args.save_dir}"
    )


if __name__ == '__main__':
    args = parse_args()
    main(args)