import torchaudio

torchaudio.datasets.LIBRISPEECH(
    root="./data",
    url="train-clean-100",
    download=True,
)

torchaudio.datasets.LIBRISPEECH(
    root="./data",
    url="test-clean",
    download=True,
)

print("Done")
