def test_torch():
    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
    except Exception as e:
        print(f"PyTorch error: {str(e)}")


def test_torchtext():
    try:
        import torchtext
        from torchtext.data.utils import get_tokenizer

        print(f"TorchText version: {torchtext.__version__}")

        # Try to create a tokenizer
        tokenizer = get_tokenizer("basic_english")
        text = "Testing torchtext functionality"
        tokens = tokenizer(text)
        print(f"Tokenization test: {tokens}")

    except Exception as e:
        print(f"TorchText error: {str(e)}")


if __name__ == "__main__":
    print("Testing PyTorch...")
    test_torch()
    print("\nTesting TorchText...")
    test_torchtext()
