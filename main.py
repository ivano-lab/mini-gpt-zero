import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Caminho do corpus
DATA_PATH = Path("data/corpus.txt")

# Hiperparâmetros iniciais
block_size = 8   # tamanho do contexto (número de caracteres anteriores)
batch_size = 4   # número de sequências por batch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. Leitura do texto
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    text = f.read()

# 2. Tokenização caractere a caractere
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocabulário: {vocab_size} caracteres")

# Mapas de conversão
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

# Funções auxiliares
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Texto codificado
data = torch.tensor(encode(text), dtype=torch.long)

print("Trecho original:")
print(text[:2000])
print("\nCodificado:")
print(encode(text[:2000]))

# 3. Criação dos batches
def get_batch(split):
    # separa entre treino e validação (90/10)
    split_idx = int(0.9 * len(data))
    data_split = data[:split_idx] if split == 'train' else data[split_idx:]

    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# Exemplo de uso
if __name__ == "__main__":
    x_batch, y_batch = get_batch('train')
    print("Entrada (x):")
    print(x_batch)
    print("Como texto:")
    for i in range(batch_size):
        print(decode(x_batch[i].tolist()), "->", decode(y_batch[i].tolist()))
