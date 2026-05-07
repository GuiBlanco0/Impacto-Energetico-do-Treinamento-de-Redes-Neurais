# SERS — Impacto Energético do Treinamento de Redes Neurais

**FIAP — 1º Ano de Ciências da Computação**
Checkpoint 2 — Energias Renováveis e Sustentabilidade

Experimento que treina uma rede neural simples no dataset MNIST com diferentes quantidades de épocas e mede o consumo de energia elétrica e as emissões de CO₂ geradas em cada cenário — conectando IA e sustentabilidade.

---

## Objetivo

Investigar a relação entre o número de épocas de treinamento de um modelo de deep learning e seu custo energético/ambiental, respondendo à pergunta: **mais treinamento sempre vale a pena?**

---

## Como funciona

O script executa quatro experimentos independentes (A, B, C e D), cada um treinando a mesma rede neural do zero com um número diferente de épocas. Para cada experimento, coleta:

- Acurácia final no conjunto de teste
- Energia consumida em kWh
- CO₂ emitido em kg (via [CodeCarbon](https://codecarbon.io))

Ao final, exibe uma tabela comparativa com todos os resultados.

---

## Experimentos

| Experimento | Épocas |
|---|---|
| A | 10 |
| B | 50 |
| C | 100 |
| D | 300 |

---

## Arquitetura da rede neural

Rede densa simples (fully connected) para classificação de dígitos manuscritos:

```
Entrada (28×28 = 784)
       ↓
  Linear → 128 → ReLU
       ↓
  Linear → 64  → ReLU
       ↓
  Linear → 10 (classes 0–9)
```

| Hiperparâmetro | Valor |
|---|---|
| Batch size | 64 |
| Learning rate | 0.001 |
| Otimizador | Adam |
| Função de perda | CrossEntropyLoss |
| Dataset | MNIST (60k treino / 10k teste) |

---

## Saída esperada

```
==================================================
Experimento: 10 épocas
==================================================
  Época   10/10 — Acurácia no teste: 0.9721

  ✔ Acurácia final : 0.9721
  ✔ Energia        : 0.000312 kWh
  ✔ CO₂ emitido    : 0.000148 kg

============================================================
                   TABELA DE RESULTADOS
============================================================
  Épocas |   Acurácia |  Energia (kWh) |     CO₂ (kg)
------------------------------------------------------------
      10 |     0.9721 |       0.000312 |     0.000148
      50 |     0.9810 |       0.001547 |     0.000734
     100 |     0.9834 |       0.003089 |     0.001466
     300 |     0.9851 |       0.009201 |     0.004369
============================================================
```

*(valores ilustrativos — os reais dependem do hardware)*

---

## Como executar

**Pré-requisitos:**

```bash
pip install torch torchvision codecarbon
```

**Executar:**

```bash
python checkpoint2.py
```

O script baixa o MNIST automaticamente na primeira execução (pasta `./data`) e salva os logs de emissão em `./emissions/emissions.csv`.

---

## Estrutura do projeto

```
.
├── checkpoint2.py        # Script principal
├── data/                 # Dataset MNIST (gerado automaticamente)
├── emissions/            # CSVs com métricas de energia e CO₂
└── README.md
```

---

## Dependências

| Biblioteca | Uso |
|---|---|
| `torch` / `torchvision` | Definição, treinamento e avaliação da rede neural |
| `codecarbon` | Medição de consumo energético e emissões de CO₂ |

---

## Contexto acadêmico

Projeto desenvolvido no contexto da disciplina de **Energias Renováveis e Sustentabilidade** da FIAP. A iniciativa SERS (Soluções em Energias Renováveis e Sustentáveis) propõe que profissionais de tecnologia compreendam o impacto ambiental das soluções que desenvolvem — inclusive modelos de inteligência artificial.
