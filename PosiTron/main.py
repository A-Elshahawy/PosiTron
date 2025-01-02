import torch
from layers.utils import subsequent_mask
from models import OptimusPrime


def test_inference():
    test_model = OptimusPrime()
    #  ? OR use ==> test_model = create_transformer(pe_type="absolute", src_vocab_size=12, tgt_vocab_size=12)
    test_model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])
    src_mask = torch.ones(1, 1, src.size(1))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_model.to(device)
    module = test_model
    module = test_model.modules
    print(module)
    memory = test_model.encode(src, src_mask)

    ys = torch.zeros(1, 1).type_as(src)
    for _ in range(src.size(1)):
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

    return name, f":: \n \t Example Untrained Model Prediction: {ys.squeeze().tolist()}"


def main():
    print(*test_inference())


if __name__ == "__main__":
    main()
