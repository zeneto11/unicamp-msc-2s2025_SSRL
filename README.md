## Adaptação de Domínio de SLM via Fine-Tuning Auto-Supervisionado com LoRA

**Proposta de projeto da disciplina MO436C — Introdução ao Aprendizado Auto-Supervisionado**
**Instituto de Computação, Universidade Estadual de Campinas (UNICAMP)**

**Equipe**

- Alejandro Núñez Arroyo — RA 299215
- Daniel da Costa Nunes Resende Neto — RA 169408
- José Augusto de Almeida Neto — RA 299218

---

### Objetivo

Este projeto tem como objetivo investigar a **adaptação de domínio** de _Small Language Models (SLM)_ por meio de **fine-tuning auto-supervisionado (SSRL)** com o método **LoRA (Low-Rank Adaptation)**.
A proposta busca avaliar o impacto de diferentes estratégias de treinamento auto-supervisionado sobre o desempenho de modelos de linguagem em tarefas de _question answering_ (Q&A) específicas de um domínio.

---

### Estrutura do Projeto

#### **1. Dados**

- Coletar textos de um domínio específico (por exemplo, **Wikipédia**).
- Gerar automaticamente (com uma LLM) ou aproveitar perguntas e respostas baseadas nesses textos.
- Dividir o conjunto de Q&A em **treino** e **teste**.

---

#### **2. SLM e SSRL**

- Selecionar uma **SLM não treinada** (ex.: Gemma, TinyLlama, etc.).
- Realizar **fine-tuning auto-supervisionado**:

  - (a) usando apenas os **textos** (masking ou language modeling).
  - (b) usando os **pares de Q&A** (auto-supervisionado sobre instruções).

- Utilizar **LoRA** para reduzir custo computacional e número de parâmetros atualizados.

---

#### **3. Avaliação**

Comparar o desempenho das seguintes variantes no conjunto de **teste (Q&A)**:

| Modelo                    | Treinamento                                   | Descrição                             |
| ------------------------- | --------------------------------------------- | ------------------------------------- |
| **SLM base**              | —                                             | Modelo original sem ajuste            |
| **SLM ajustada (textos)** | Fine-tuning auto-supervisionado com textos    | Avalia impacto do domínio             |
| **SLM ajustada (Q&A)**    | Fine-tuning auto-supervisionado com pares Q&A | Avalia capacidade de adaptação direta |

---

### Como Preparar Ambiente

```bash
# Instalar dependências
poetry install

# Ativar ambiente virtual
poetry shell
```

---
