# DharmaOCR POC — estudo e comparação de OCRs

POC para estudar o **Dharma OCR** e compará-lo com outros OCRs do mercado na figura de mérito do paper *DharmaOCR: Specialized Small Language Models for Structured OCR* ([arXiv:2604.14314](https://arxiv.org/abs/2604.14314)), usando o benchmark [`Dharma-AI/DharmaOCR-Benchmark`](https://huggingface.co/datasets/Dharma-AI/DharmaOCR-Benchmark).

## Notebooks

| Notebook | O que faz |
|---|---|
| `01_setup_dataset_metric_reader.ipynb` | Baixa o dataset, monta a figura de mérito e a chamada ao leitor (API HTTP do Dharma, `POST /v1/ocrs/`). |
| `02_kfold_benchmark_comparison.ipynb` | Valida os números do paper via **k-fold** (média ± IC95) e compara Dharma × Baidu Unlimited-OCR × GLM-OCR na *mesma* métrica, com teste de significância pareado. |

## Figura de mérito (fiel ao `evaluation.ipynb` oficial do benchmark)

```
benchmark_score = (Levenshtein_Ratio + BLEU) / 2
```

- **Levenshtein_Ratio** = `1 − distance / max(len)` (0.0 quando ambos vazios).
- **BLEU** = NLTK `sentence_bleu([gt.split()], [pred.split()], SmoothingFunction().method1)` (BLEU-4).
- **Ground-truth** = coluna `assistant` (JSON) normalizada por `PLAIN_TEXT`.
- Métrica secundária: *text degeneration rate* (limite de tokens **e** repetição dos últimos 15 chars ≥4×).

## Modelos comparados

| Modelo | Licença | Acesso |
|---|---|---|
| Dharma-OCR full / lite | open-weights (LITE público, FULL *gated*) | API HTTP `/v1/ocrs/` ou vLLM |
| Baidu Unlimited-OCR | MIT | vLLM self-host (OpenAI-compatible) |
| GLM-OCR | MIT | vLLM self-host, ou API hospedada da Z.ai |

## Como rodar (Google Colab, GPU T4)

1. Abra o notebook no Colab com runtime **GPU**.
2. Rode a célula de dependências (inclui `vllm` para os modelos abertos).
3. Configure as chaves/endpoints por variável de ambiente (`DHARMA_API_KEY`, `GLM_OCR_URL`, `BAIDU_OCR_URL`, …) — **nunca commite chaves**.
4. Suba **um modelo aberto por vez** via `start_vllm(...)` (a T4 comporta um de cada vez), rode `run_inference(...)` e `stop_vllm(...)`.
5. Rode as seções de k-fold, comparação e significância.

> Os subconjuntos ESTER-Pt/Legal/BRESSAY não vêm rotulados por linha no release, então o k-fold é sobre os 496 documentos como um todo.
