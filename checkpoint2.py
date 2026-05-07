"""
FIAP - 1º Ano de Ciências da Computação
SERS – Soluções em Energias Renováveis e Sustentáveis
Checkpoint 2 – Energias Renováveis e Sustentabilidade
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from codecarbon import EmissionsTracker

# ─────────────────────────────────────────
# 1. CONFIGURAÇÕES GERAIS
# ─────────────────────────────────────────
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS_LIST = [10, 50, 100, 300]   # Experimentos A, B, C e D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}\n")

# ─────────────────────────────────────────
# 2. DATASET MNIST
# ─────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))   # média e desvio padrão do MNIST
])

train_dataset = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

# ─────────────────────────────────────────
# 3. ARQUITETURA DA REDE NEURAL
# ─────────────────────────────────────────
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)

# ─────────────────────────────────────────
# 4. FUNÇÕES DE TREINAMENTO E AVALIAÇÃO
# ─────────────────────────────────────────
def train(model, loader, optimizer, criterion):
    model.train()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            _, predicted = torch.max(output, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total

# ─────────────────────────────────────────
# 5. EXPERIMENTOS
# ─────────────────────────────────────────
import os
os.makedirs("./emissions", exist_ok=True)

results = []

for epochs in EPOCHS_LIST:
    print(f"{'='*50}")
    print(f"Experimento: {epochs} épocas")
    print(f"{'='*50}")

    # Reinicia o modelo e otimizador a cada experimento
    model = SimpleNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Inicia o rastreador de emissões
    tracker = EmissionsTracker(
        project_name=f"MNIST_{epochs}_epocas",
        output_dir="./emissions",
        log_level="error"       # oculta logs internos do CodeCarbon
    )
    tracker.start()

    # Treinamento
    for epoch in range(1, epochs + 1):
        train(model, train_loader, optimizer, criterion)
        if epoch % max(1, epochs // 5) == 0 or epoch == epochs:
            acc = evaluate(model, test_loader)
            print(f"  Época {epoch:>4}/{epochs} — Acurácia no teste: {acc:.4f}")

    # Para o rastreador e coleta métricas
    emissions = tracker.stop()   # retorna kg de CO₂

    # Lê o kWh do arquivo CSV gerado pelo CodeCarbon
    import csv
    energy_kwh = None
    csv_path = f"./emissions/emissions.csv"
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
            if rows:
                energy_kwh = float(rows[-1]["energy_consumed"])   # kWh

    accuracy = evaluate(model, test_loader)

    results.append({
        "epocas": epochs,
        "acuracia": accuracy,
        "energia_kwh": energy_kwh,
        "co2_kg": emissions
    })

    print(f"\n  ✔ Acurácia final : {accuracy:.4f}")
    print(f"  ✔ Energia        : {energy_kwh:.6f} kWh" if energy_kwh else "  ✔ Energia        : N/A")
    print(f"  ✔ CO₂ emitido    : {emissions:.6f} kg\n")

# ─────────────────────────────────────────
# 6. TABELA DE RESULTADOS
# ─────────────────────────────────────────
print("\n" + "="*60)
print(f"{'TABELA DE RESULTADOS':^60}")
print("="*60)
print(f"{'Épocas':>8} | {'Acurácia':>10} | {'Energia (kWh)':>14} | {'CO₂ (kg)':>12}")
print("-"*60)
for r in results:
    energia = f"{r['energia_kwh']:.6f}" if r['energia_kwh'] else "N/A"
    print(f"{r['epocas']:>8} | {r['acuracia']:>10.4f} | {energia:>14} | {r['co2_kg']:>12.6f}")
print("="*60)