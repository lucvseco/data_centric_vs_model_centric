# experiment_Pipeline — Experimento de Previsão de Séries Temporais

Desenho experimental hierárquico que compara modelos de previsão de séries temporais univariadas sob diferentes estratégias de transformação, otimizado com Optuna.

---

## Estrutura do notebook

O notebook `experiment_Pipeline.ipynb` está organizado em três seções executáveis de forma independente.

### 1. Pré-processamento

Lê arquivos de `data/` (CSV ou Excel), detecta automaticamente a coluna de data (tenta a 1ª coluna; se falhar, a 2ª) e mantém a última coluna como variável alvo. Salva pares `[date, y]` limpos e ordenados em `data_processed/`.

Para cada série, imprime um diagnóstico completo:

- **Dimensão**: número de observações, intervalo temporal, frequência inferida (diária, semanal, mensal…)
- **Descritivas**: média, mediana, desvio padrão, coeficiente de variação, assimetria, curtose, presença de zeros/negativos
- **Estacionariedade**: ADF + KPSS combinados com conclusão qualitativa (estacionária / não estacionária / tendência determinística / estacionária por diferenças)
- **Sazonalidade e tendência**: período candidato via ACF, forças de tendência e sazonalidade (F_T, F_S) via decomposição STL, razão de ruído
- **Outliers**: contagem pela regra IQR
- **Diagnóstico final**: notas automáticas sobre transformações e modelos recomendados para o perfil da série

### 2. Análise Exploratória (EDA)

Gera dois painéis por série a partir dos arquivos em `data_processed/`, salvos em `results/eda/`:

**Painel principal** (`eda_<nome>.png`):
- Série temporal completa
- Histograma com estimativa de densidade (KDE)
- Média e desvio padrão móveis (estacionariedade visual)
- Decomposição STL: componentes de tendência e sazonalidade
- ACF e PACF

**Painel sazonal** (`sazonalidade_<nome>.png`):
- Boxplot por ano (distribuição entre anos)
- Boxplot por mês (padrão sazonal intra-anual, quando há ≥ 2 anos)

### 3. Experimento

#### Modelos avaliados

| Modelo | Estratégia de previsão |
|--------|------------------------|
| `linear` | Regressão linear com janela de lags |
| `arima` | ARIMA(p,d,q) via statsmodels |
| `xgboost` | XGBoost com janela de lags |
| `lstm` | LSTM com early stopping e weight decay |
| `transformer` | Transformer encoder com early stopping e weight decay |

#### Grupos de transformação

Cada modelo é otimizado dentro de cada um dos quatro grupos de pré-transformação:

| Grupo | Transformação aplicada |
|-------|------------------------|
| A | Apenas normalização (z-score / robust / minmax) |
| B | Diferenciação sazonal + normalização |
| C | Log ou Box-Cox + normalização |
| D | Log ou Box-Cox + diferenciação sazonal + normalização |

#### Pipeline de transformação

A classe `TransformPipeline` aplica as transformações na ordem: distribuição → diferenciação sazonal → normalização. A inversão é feita na ordem reversa para garantir que todas as métricas sejam calculadas na escala original.

O lambda do Box-Cox e os parâmetros do scaler são estimados **exclusivamente no conjunto de treino** de cada dobra, evitando vazamento de dados.

#### Otimização (Optuna)

Para cada combinação `(dataset, modelo, grupo)` é rodado um estudo Optuna independente:

- **Validação**: `TimeSeriesSplit` com 3 dobras sobre `train + val`
- **Métrica de otimização**: RMSE médio entre dobras (escala original)
- **Sampler**: TPE com semente fixa (`RANDOM_STATE = 42`)
- **Pruner**: MedianPruner (descarta trials ruins após cada dobra)
- **Paralelismo**: `n_jobs = N_JOBS` (padrão: metade dos cores da máquina)
- **Sanity check**: previsões além de 20 desvios-padrão do treino são descartadas como numericamente instáveis

Após a otimização, a melhor configuração é reajustada em `train + val` completo e avaliada no **test set**, que permanece intocado durante toda a otimização.

#### Métricas reportadas no test

MAE, RMSE, MAPE, sMAPE e MASE (normalizado pelo naive sazonal).

#### Resultados

Dois CSVs são salvos em `results/`:

- `validation_best.csv` — melhor RMSE de validação cruzada por combinação, com hiperparâmetros e número de trials podados
- `test_results.csv` — métricas completas no test set para cada combinação

---

## Estrutura de pastas

```
.
├── data/                   # Arquivos de entrada (CSV ou Excel)
├── data_processed/         # Séries processadas [date, y]
├── results/
│   ├── eda/                # Gráficos da análise exploratória
│   ├── validation_best.csv # Resultados da otimização Optuna
│   └── test_results.csv    # Métricas no test set
└── v3paralelo.ipynb
```

## Dependências principais

```
numpy pandas scipy statsmodels matplotlib seaborn
scikit-learn xgboost torch optuna
```

## Como usar

1. Coloque os arquivos de dados em `data/` (CSV ou Excel com coluna de data e coluna alvo)
2. Execute a **Seção 1** para pré-processar e inspecionar as séries
3. Execute a **Seção 2** para gerar os gráficos exploratórios
4. Ajuste o dicionário `datasets` na **Seção 3** e execute para rodar o experimento
5. Os resultados ficam em `results/`
