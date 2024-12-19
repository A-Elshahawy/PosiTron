import sys

import torch
from factories.ModelFactory import create_transformer

# from .models import make_model
from layers.utils import subsequent_mask

sys.path.append(r"D:\Projects\transformer")


def test_inference(pe_type="absolute"):
    # test_model = make_model(11, 11, 2)

    test_model = create_transformer(pe_type, 11, 11, n_heads=2, d_model=512)
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

    return f"Example Untrained Model Prediction: {ys.squeeze().tolist()}"


def main():
    pe_type = ["absolute", "relative", "rotary", "alibi"]
    for p in pe_type:
        print(
            " " * 20
            + f"\033[1;32mTesting \033[0;33m{p.capitalize()}\033[1;32m Transformer inference!\033[0m"
        )
        for _ in range(10):
            print(test_inference(pe_type=p))


if __name__ == "__main__":
    main()
