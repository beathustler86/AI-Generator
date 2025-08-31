# src/nodes/my_model_defs.py

# Align imports to package root (src.*) to avoid relative ambiguity.
from src.nodes.cosmos.model import GeneralDIT as CosmosModel
from src.nodes.cosmos.vae import CosmosVAE
from src.nodes.cosmos.encoder import CosmosEncoder  # adjust if actual filename differs

class CosmosEncoder:
    def __init__(self):
        self.tokenizer = Tokenizer3D()

    def encode(self, text: str):
        return self.tokenizer(text)
