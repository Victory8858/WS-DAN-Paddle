# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import paddle
from models.wsdan import WSDAN


def parse_args():
    parser = argparse.ArgumentParser(description='Fined-Grained Classification')
    parser.add_argument("--model", default="bird", type=str, help="bird, car, aircraft, bird_tiny")
    parser.add_argument("--input-size", default=(448, 448), type=tuple)
    parser.add_argument('--save-dir', default='output', type=str, help='The directory for saving the exported model')

    return parser.parse_args()

def main(args):
    # model
    if args.model == 'bird':
        num_classes = 200
    elif args.model == 'car':
        num_classes = 196
    elif args.model == 'aircraft':
        num_classes = 100
    elif args.model == 'bird_tiny':
        num_classes = 5
    paddle.disable_static()
    model = WSDAN(num_classes=num_classes, num_attentions=32, net_name='inception_mixed_6e', pretrained=False)
    model_path = os.path.join('FGVC', args.model, args.model) + '_model.pdparams'
    print("model_path: ", model_path)
    net_state_dict = paddle.load(model_path)
    model.set_dict(net_state_dict)
    print('Loaded trained params of model successfully.')
    model.eval()

    model = paddle.jit.to_static(model, input_spec=[paddle.static.InputSpec(shape=[1, 3, 448, 448], dtype="float32")])
    # save inference model
    paddle.jit.save(model, os.path.join(args.save_dir, "inference"))
    print(
        f"inference model  have been saved into {args.save_dir}"
    )


if __name__ == '__main__':
    args = parse_args()
    main(args)
