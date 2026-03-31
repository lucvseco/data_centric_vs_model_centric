# Previsão de Arrecadação Previdenciária: Data-Centric vs. Model-Centric

> **Notebook:** `tese_previdencia_data_vs_model_centric.ipynb`
> **Contexto:** Experimento comparativo para pesquisa de tese — BRACIS
> **Dados:** Arrecadação mensal da Base DBP (ago/2003 – ago/2024 · 253 observações)

---

## Objetivo

Investigar qual estratégia produz melhores previsões de séries temporais de arrecadação previdenciária: investir no **pré-processamento e engenharia de features** (Data-Centric) ou investir na **otimização automática de hiperparâmetros** (Model-Centric), mantendo o mesmo conjunto de algoritmos base (SARIMA, XGBoost e LSTM) nos dois cenários.

---

## Estrutura do Notebook

O notebook é dividido em **4 etapas**, cada uma precedida por um cabeçalho Markdown, e pode ser executado do início ao fim via *Run All* sem erros de variáveis.

```
Cell 0  [MD]    ## Configurações e Imports
Cell 1  [Code]  Imports unificados + variáveis globais (BASE_PATH, OUT, hiperparâmetros)
Cell 2  [MD]    ## Etapa 1: Análise Exploratória e Decomposição
Cell 3  [Code]  EDA completa da série
Cell 4  [MD]    ## Etapa 2: Cenário Model-Centric (LSTM Robusta)
Cell 5  [Code]  Pré-processamento mínimo + Optuna (50 trials por modelo)
Cell 6  [MD]    ## Etapa 3: Cenário Data-Centric (Engenharia de Features + LSTM Simples)
Cell 7  [Code]  Box-Cox + feature engineering + modelos com hiperparâmetros padrão
Cell 8  [MD]    ## Etapa 4: Comparação de Resultados e Conclusão
Cell 9  [Code]  Métricas, gráficos comparativos e análise crítica
```

---

## Etapas em Detalhe

### Etapa 1 — Análise Exploratória e Decomposição
- Carregamento e preparação do índice mensal (freq `MS`)
- **Decomposição STL** (period=12, robust=True): separa tendência, sazonalidade e resíduos
- **Teste ADF** (Augmented Dickey-Fuller): verificação de estacionaridade e primeira diferenciação se necessário
- **ACF / PACF** com identificação de lags significativos a 95%
- **Detecção de outliers** por Z-score ≥ 3
- **Divisão cronológica** treino / teste: últimos 24 meses como teste

### Etapa 2 — Cenário Model-Centric
**Premissa:** pré-processamento mínimo; o algoritmo precisa atingir seu potencial máximo via busca automática.

| Passo | Detalhe |
|-------|---------|
| Pré-processamento | Apenas `RobustScaler` na série original |
| Features | 12 lags (`lag_1` … `lag_12`) — sem transformações extras |
| Validação interna | Hold-out dos últimos 12 meses do treino (para o Optuna) |
| Otimizador | **Optuna TPE** · 50 trials por modelo · `seed=42` |

Espaços de busca por modelo:

| Modelo | Hiperparâmetros otimizados |
|--------|---------------------------|
| **SARIMA** | `p,d,q` ∈ [0,2] · `P,D,Q` ∈ [0,1] · `S=12` |
| **XGBoost** | `learning_rate` (log 1e-3→0.1) · `max_depth` (3→10) · `subsample` (0.5→1.0) · `n_estimators` (100→500) |
| **LSTM** | `units` (32→256) · `dropout` (0.1→0.5) · `learning_rate` Adam (log 1e-4→1e-2) |

Após a busca, treino final sobre **todo** o conjunto de treino com os melhores hiperparâmetros. Os tempos de busca (*search time*) e de treino final são registrados separadamente.

**Artefatos salvos:** `mc_artifacts.pkl` · `mc_lstm_model.keras` · `mc_best_params.txt`

### Etapa 3 — Cenário Data-Centric
**Premissa:** hiperparâmetros padrão (simples); os dados é que precisam carregar a informação.

| Passo | Detalhe |
|-------|---------|
| Transformação | **Box-Cox** (λ ótimo estimado no treino, aplicado ao teste sem data leakage) |
| Features | 12 lags + diferenciação sazonal `lag_t − lag_{t-12}` + médias móveis `MA(3)` e `MA(12)` |
| Normalização | `RobustScaler` separado para X e y |
| SARIMA | Ordem fixa `(1,1,1)(1,1,1,12)`, fit na série Box-Cox |
| XGBoost | `n_estimators=100`, `max_depth=3`, `lr=0.1` |
| LSTM | 50 unidades · `dropout=0.2` · `EarlyStopping(patience=15)` · input 3D com lags + features extras replicadas por timestep |

**Artefatos salvos:** `dc_artifacts.pkl`

### Etapa 4 — Comparação de Resultados e Conclusão
- **Baseline:** Naïve Sazonal — `ŷ(t) = y(t−12)` (benchmark mínimo a superar)
- **Métricas:** MAE · RMSE · MAPE · sMAPE · **Theil-U2** (razão ao RMSE do Naïve)
- **Tabela geral** com todos os 7 modelos (1 baseline + 3 MC + 3 DC)
- **Síntese por abordagem** — média das métricas dos 3 modelos

**Gráficos gerados** (salvos em `OUT`):

| Arquivo | Conteúdo |
|---------|----------|
| `11_comparacao_previsoes.png` | Previsões vs. Real para os 3 algoritmos (Naïve / MC / DC) |
| `12_metricas_comparacao.png` | Barras agrupadas: MAE, RMSE, MAPE, sMAPE por modelo |
| `13_erros_mensais.png` | Erros absolutos mês a mês com destaque para dezembro |
| `14_theil_u2.png` | Theil-U2 horizontal — ganho relativo sobre o Naïve |

Inclui **análise crítica textual** cobrindo: contexto da série previdenciária, tratamento da sazonalidade S=12, custo-benefício computacional da busca Optuna, e erros nos meses de pico (dezembro).

---

## Dados

| Atributo | Valor |
|----------|-------|
| Arquivo | `Base_DBP.xlsx` |
| Colunas | `data` (datetime), `valores` (R$ arrecadação mensal) |
| Período | Agosto/2003 → Agosto/2024 |
| Observações | 253 meses |
| Treino | 229 observações |
| Teste | 24 observações (últimos 2 anos) |
| Sazonalidade dominante | Anual (S=12) — pico em dezembro (13º salário) |

---

## Requisitos

```
python       >= 3.10
pandas       >= 2.0
numpy        >= 1.24
matplotlib   >= 3.7
scipy        >= 1.11
scikit-learn >= 1.3
statsmodels  >= 0.14
xgboost      >= 2.0
optuna       >= 3.0
tensorflow   >= 2.13
```

Instalação rápida:

```bash
pip install pandas numpy matplotlib scipy scikit-learn statsmodels xgboost optuna tensorflow
```

---

## Como Executar

1. Certifique-se de que `Base_DBP.xlsx` está em `C:/Users/lucas/bracis/` (ou ajuste `BASE_PATH` na Célula 1).
2. Configure `OUT` para o diretório de saída desejado (padrão: `/mnt/user-data/outputs`).
3. Execute *Kernel → Restart & Run All*.

A execução é totalmente linear — cada etapa depende apenas das variáveis definidas nas etapas anteriores. A Etapa 2 (Optuna LSTM, 50 trials) é a mais demorada.

---

## Variáveis Globais (Célula 1)

| Variável | Valor padrão | Descrição |
|----------|-------------|-----------|
| `BASE_PATH` | `"C:/Users/lucas/bracis/Base_DBP.xlsx"` | Caminho para os dados |
| `OUT` | `"/mnt/user-data/outputs"` | Diretório de saída para gráficos e artefatos |
| `N_TEST` | `24` | Meses reservados para teste |
| `N_VAL` | `12` | Meses de validação interna (Optuna) |
| `WINDOW` / `WINDOW_SIZE` | `12` | Tamanho da janela de lags |
| `N_TRIALS` | `50` | Número de trials Optuna por modelo |
| `RANDOM_SEED` | `42` | Semente global de reprodutibilidade |

---

## Artefatos de Saída

```
OUT/
├── 01_decomposicao_stl.png       # Decomposição STL (tendência, sazonalidade, resíduos)
├── 02_acf_pacf.png               # ACF e PACF da série (diferenciada se necessário)
├── 03_picos_zscore.png           # Série com outliers Z-score ≥ 3 destacados
├── 04_train_test_split.png       # Divisão cronológica treino/teste
├── 05_previsoes_etapa2_mc.png    # Previsões Model-Centric vs Real (por algoritmo)
├── 06_optuna_historico.png       # Convergência Optuna (scatter + melhor acumulado)
├── 07_tempos_etapa2.png          # Search Time vs Train Final por modelo
├── 08_previsoes_etapa3_dc.png    # Previsões Data-Centric vs Real
├── 09_lstm_learning_curve.png    # Curva de aprendizado LSTM (treino/validação)
├── 10_xgb_feature_importance.png # Importância de features XGBoost (Data-Centric)
├── 11_comparacao_previsoes.png   # Painel comparativo final (3 modelos × 3 abordagens)
├── 12_metricas_comparacao.png    # Barras de métricas comparativas
├── 13_erros_mensais.png          # Erros absolutos mensais com destaque dezembro
├── 14_theil_u2.png               # Theil-U2 relativo ao Naïve Sazonal
├── mc_artifacts.pkl              # Artefatos Model-Centric (modelos, previsões, tempos)
├── mc_lstm_model.keras           # Modelo LSTM Model-Centric serializado
├── mc_best_params.txt            # Melhores hiperparâmetros encontrados pelo Optuna
└── dc_artifacts.pkl              # Artefatos Data-Centric (modelos, previsões, scaler)
```

---

## Hipótese Central

> Para séries de arrecadação previdenciária — caracterizadas por **forte sazonalidade anual regular**, **tendência crescente gradual** e **base de dados moderada** (< 300 observações) — a abordagem **Data-Centric** oferece melhor custo-benefício: a combinação Box-Cox + features sazonais explícitas eleva o desempenho de modelos simples sem requerer busca hiperparamétrica exaustiva. A abordagem **Model-Centric** agrega valor principalmente ao SARIMA (identificação automática da ordem ótima), mas tem retorno marginal para XGBoost e LSTM quando o pré-processamento é mínimo.
