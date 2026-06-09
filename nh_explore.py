import inspect
import torch
import torch.nn as nn
from neuralhydrology.modelzoo.cudalstm import CudaLSTM

# model zoo
model_names = [
    "CudaLSTM", "EALSTM", "EmbCudaLSTM", "CustomLSTM", "GRU",
    "ODELSTM", "MCLSTM", "Transformer", "Mamba", "XLSTM",
    "ARLSTM", "MTSLSTM", "SequentialForecastLSTM",
    "StackedForecastLSTM", "MultiHeadForecastLSTM", "HandoffForecastLSTM"
]
print(f"NeuralHydrology Model Zoo ({len(model_names)} architectures):")
for name in model_names:
    print(f"  {name}")

# CudaLSTM source
print("\nCudaLSTM.__init__ source:")
print(inspect.getsource(CudaLSTM.__init__))

# environment
print(f"PyTorch: {torch.__version__}")
print(f"MPS: {torch.backends.mps.is_available()}")
print(f"CUDA: {torch.cuda.is_available()}")

# encoder-decoder LSTM matching Nearing et al. (2024)
encoder = nn.LSTM(input_size=15, hidden_size=256, num_layers=1, batch_first=True)
decoder = nn.LSTM(input_size=6, hidden_size=256, num_layers=1, batch_first=True)
transfer_h = nn.Sequential(nn.Linear(256, 256), nn.Tanh())
transfer_c = nn.Linear(256, 256)
output_head = nn.Linear(256, 3)

components = {
    "Encoder LSTM": encoder, "Decoder LSTM": decoder,
    "Transfer (h)": transfer_h, "Transfer (c)": transfer_c,
    "Output head": output_head
}
total = 0
print("\nParameter counts:")
for name, module in components.items():
    n = sum(p.numel() for p in module.parameters())
    total += n
    print(f"  {name:<20s} {n:>10,d}")
print(f"  {'Total':<20s} {total:>10,d}")

# forward pass
x_enc = torch.randn(4, 365, 15)
x_dec = torch.randn(4, 7, 6)
enc_out, (h_n, c_n) = encoder(x_enc)
dec_out, _ = decoder(x_dec, (transfer_h(h_n), transfer_c(c_n)))
pred = output_head(dec_out)

print(f"\nForward pass:")
print(f"  Encoder input:  {list(x_enc.shape)}")
print(f"  Encoder output: {list(enc_out.shape)}")
print(f"  Decoder input:  {list(x_dec.shape)}")
print(f"  Predictions:    {list(pred.shape)}")
print(f"  -> batch=4, days=7, params=3")