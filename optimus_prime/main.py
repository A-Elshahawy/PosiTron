import sys

import torch

from .models import make_model
from .utils import subsequent_mask

sys.path.append(r"D:\Projects\transformer")


def test_inference():
    test_model = make_model(11, 11, 2)
    test_model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)

    memory = test_model.encode(src, src_mask)
    ys = torch.zeros(1, 1).type_as(src)
    for i in range(9):
        out = test_model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        pred_proba = test_model.generator(out[:, -1])
        _, next_word = torch.max(pred_proba, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )

    print("Example Untrained Model Prediction:", ys)


def main():
    print(" " * 20, "\033[1;32mTesting transformer infrence!\033[0m")
    for _ in range(10):
        test_inference()


if __name__ == "__main__":
    main()
