from tokenizer import DataProcessing
from embedding import Embedder
from model import ClassifyModel
import torch

device = torch.device("cpu")
ds = DataProcessing()
emb = Embedder(device)
model = ClassifyModel().to(device)

def predict_model(message):
    tokens = ds.convert_data_to_token(message, 256)
    tokens = tokens.to(device)
    token_emb = emb.forward(tokens).to(device)
    model = ClassifyModel().to(device)
    model.eval()
    out = model(token_emb).squeeze()
    return out.tolist()