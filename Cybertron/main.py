import random
import sys

import torch
from layers.utils import subsequent_mask
from models.model import Bumblebee, Megatron, OptimusPrime, Starscream

sys.path.append(r"D:\Projects\transformer")


def test_inference():
    test_model = Starscream()
    test_model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])
    src_mask = torch.ones(1, 1, src.size(1))

    memory = test_model.encode(src, src_mask)

    ys = torch.zeros(1, 1).type_as(src)
    for i in range(src.size(1)):
        out = test_model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        pred_proba = test_model.generator(out[:, -1])
        _, next_word = torch.max(pred_proba, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    name = f"\033[1;32m\033[0;33m{test_model.__class__.__name__.capitalize()}\033[1;32m Transformer inference!\033[0m"
    return name, f":: Example Untrained Model Prediction: {ys.squeeze().tolist()}"


def main():
    pe_type = ["absolute", "relative", "rotary", "alibi"]
    for _ in range(len(pe_type)):
        p = random.choice(pe_type)
        print(
            " " * 20
            + f"\033[1;32mTesting \033[0;33m{p.capitalize()}\033[1;32m Transformer inference!\033[0m"
        )
        for _ in range(10):
            print(test_inference())


if __name__ == "__main__":
    # main()
    print(*test_inference())
