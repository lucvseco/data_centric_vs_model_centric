# Forecasting Experiments: Data-Centric vs. Model-Centric + Baselines Modernos

> **Notebook:** `forecasting_experiments.ipynb`
> **Contexto:** Experimento comparativo de forecasting univariado mensal para pesquisa
> **Escopo:** Framework experimental para comparação de abordagens em séries temporais **univariadas** e **mensais**

---

## Introdução

Este repositório organiza um experimento comparativo de previsão de séries temporais mensais sob duas filosofias de desenvolvimento de modelos:

- **Data-Centric**, em que o ganho esperado vem principalmente da representação dos dados
- **Model-Centric**, em que o ganho esperado vem principalmente da adaptação e do tuning dos algoritmos

Como referência adicional, o protocolo incorpora **baselines modernos** para impedir que a comparação se reduza a uma disputa interna entre pipelines mais complexos. A intenção do experimento não é apenas produzir previsões, mas documentar de forma reproduzível **como diferentes escolhas metodológicas se comportam sob o mesmo contrato de dados, o mesmo horizonte de teste e o mesmo critério de avaliação**.

Este `README` deve ser lido como uma **introdução técnica ao desenho experimental**. Ele explica o que o notebook faz, quais hipóteses metodológicas estão embutidas no código e quais restrições precisam ser respeitadas para que a execução faça sentido.

---

## Objetivo

Comparar, no mesmo problema de previsão mensal, três frentes metodológicas:

- **Data-Centric**: maior investimento em transformações e engenharia de atributos
- **Model-Centric**: menor intervenção no sinal e maior investimento em tuning
- **Baselines modernos**: modelos de referência para calibrar ganho real de desempenho

O notebook foi estruturado para responder não apenas **qual modelo vence no holdout**, mas também **qual abordagem entrega melhor equilíbrio entre acurácia, robustez temporal e custo computacional**.

O código foi organizado para servir como **guia técnico reutilizável** para outras bases com o mesmo contrato de entrada.

Em síntese, o experimento busca comparar:

- o efeito de enriquecer o sinal antes do treinamento
- o efeito de aumentar o esforço de tuning sobre uma representação mais simples
- o custo de cada escolha em tempo e complexidade
- a estabilidade dos pipelines ao longo do tempo

---

## Hipótese Metodológica

A motivação central do experimento é a seguinte:

> em séries mensais univariadas com sazonalidade anual marcante e volume histórico moderado, o desempenho final pode depender menos de modelos cada vez mais sofisticados e mais da forma como o sinal é preparado, validado e comparado.

Por isso, o notebook foi desenhado para separar com clareza:

- ganho por **transformação e feature engineering**
- ganho por **busca de hiperparâmetros**
- ganho real sobre **baselines competitivos**
- robustez observada no **holdout** e na **validação cruzada temporal**

---

## Requisito Crítico da Base

Este experimento foi desenhado para funcionar com bases que respeitem simultaneamente estas duas condições:

- **série univariada**: apenas uma variável alvo a ser prevista ao longo do tempo
- **frequência mensal**: observações organizadas em passos mensais regulares

Em termos práticos, a base usada no notebook deve seguir o formato:

| Coluna | Tipo | Papel |
|--------|------|-------|
| data | datetime | índice temporal mensal |
| valores | numérica | série alvo univariada |

Se a base não for univariada ou não for mensal, partes importantes do pipeline deixam de fazer sentido metodológico, por exemplo:

- sazonalidade fixa `S=12`
- lags desenhados para dinâmica mensal
- naive sazonal com defasagem de 12 períodos
- decomposição e avaliação por horizonte pensadas para meses
- comparação justa entre pipelines no mesmo contrato temporal

Em outras palavras: este repositório **não é um framework genérico para qualquer série temporal**. Ele foi construído para **forecasting univariado mensal**.

Essa restrição não é apenas técnica; ela é parte da própria validade do experimento.

---

## Desenho Experimental

O experimento foi organizado para que todas as abordagens sejam avaliadas sob o mesmo protocolo:

- mesma série alvo
- mesma divisão treino/teste
- mesmo horizonte de holdout
- mesma sazonalidade de referência
- mesmo conjunto de métricas
- mesma camada final de consolidação e persistência

Isso reduz o risco de comparações assimétricas entre pipelines e torna mais fácil interpretar se o ganho observado vem da abordagem, do modelo ou do protocolo de avaliação.

---

## Estrutura do Notebook

O notebook está organizado em **10 seções principais** e pode ser executado de forma sequencial. O fluxo central do experimento cobre: preparação do ambiente, carregamento da base, EDA, pipelines `Data-Centric` e `Model-Centric`, baselines, avaliação final, interpretação crítica, validação cruzada temporal e dashboards complementares.

```text
Seção 1   Ambiente do Notebook
Seção 2   Carregamento da Base
Seção 3   Análise Exploratória e Decomposição
Seção 4   Abordagem Data-Centric
Seção 5   Abordagem Model-Centric
Seção 6   Baselines Modernos
Seção 7   Avaliação e Comparação Final
Seção 8   Validação Cruzada Temporal - Expanding Window
Seção 9  Dashboards e Análises Complementares
```

---

## Etapas em Detalhe

### Seção 1 - Ambiente do Notebook
- Configuração central do experimento
- Definição de `SEED`, `TEST_HORIZON`, `SEASONAL_PERIOD`, `DC_WINDOW` e caminhos
- Inicialização de utilitários, métricas e rotinas de persistência
- Núcleo de previsão recursiva para os dois pipelines

### Seção 2 - Carregamento da Base
- Leitura da base configurada em `FILE_NAME`
- Padronização do índice temporal mensal
- Definição dos conjuntos de treino e teste
- Holdout externo com **12 meses**

### Seção 3 - Análise Exploratória e Decomposição
- Inspeção inicial da série
- Visualização do split treino/teste
- **Decomposição STL**
- Gráficos de apoio para entender tendência, sazonalidade e resíduos

### Seção 4 - Abordagem Data-Centric
**Premissa:** enriquecer a representação do sinal para que modelos relativamente simples extraiam mais estrutura da série.

Principais componentes:

| Bloco | Descrição |
|------|-----------|
| Transformação | **Box-Cox** ajustada exclusivamente no treino |
| Features | lags, atributos sazonais e atributos causais derivados |
| Escalonamento | `StandardScaler` ajustado apenas no treino |
| LSTM | arquitetura **dual-input** separando componentes sequenciais e estáticas |

Modelos executados:

| Modelo | Rótulo nos artefatos |
|--------|-----------------------|
| SARIMA | `DC_SARIMA` |
| XGBoost | `DC_XGB` |
| LSTM dual-input | `DC_LSTM` |

### Seção 5 - Abordagem Model-Centric
**Premissa:** manter o sinal mais próximo da série original e concentrar o esforço na capacidade adaptativa dos modelos.

Principais componentes:

| Bloco | Descrição |
|------|-----------|
| Features | **lags puros** (`lag_1 ... lag_W`) |
| Normalização | `RobustScaler` sobre a série do pipeline |
| SARIMA | busca de hiperparâmetros por **AIC** |
| XGBoost | tuning com **Optuna TPE** |
| LSTM | tuning com **Optuna** e split temporal explícito em 3 vias |

Modelos executados:

| Modelo | Rótulo nos artefatos |
|--------|-----------------------|
| SARIMA | `SARIMA-MC` |
| XGBoost | `XGBoost-MC` |
| LSTM simples | `LSTM-MC` |

### Seção 6 - Baselines Modernos
Bloco de referência para comparar os pipelines principais com modelos largamente usados em forecasting.

| Modelo | Tipo | Observação |
|--------|------|------------|
| `Prophet` | decomposição com sazonalidade anual | sem tuning |
| `Theta` | baseline estatístico moderno | com cadeia de fallback |
| `ETS` | exponential smoothing com seleção por AIC | sem tuning manual |
| `SNaive` | naive sazonal | usado como benchmark nas métricas consolidadas |

### Seção 7 - Avaliação e Comparação Final
- Consolidação das previsões disponíveis
- Comparação justa no **holdout principal**
- Tabela completa de métricas para todos os modelos
- Média por abordagem
- Melhor modelo por métrica
- Análise por **bandas de horizonte**
- Persistência dos artefatos consolidados

Métricas calculadas:

- `MAE`
- `RMSE`
- `MAPE`
- `sMAPE`
- `U2` (Theil's U2, relativo ao naive sazonal)
- `MASE`


### Seção 8 - Validação Cruzada Temporal
**Expanding window** para avaliar robustez temporal dos pipelines ao longo de múltiplos folds.

Características centrais:

| Item | Valor / Regra |
|------|----------------|
| Estratégia | `expanding window` |
| Treino mínimo inicial | `36` observações |
| Horizonte por fold | `12` meses |
| Step | `12` meses |
| Pipeline DC | sem tuning por fold, por design |
| Pipeline MC | modo `nested` ou `pragmatic`, conforme `CV_CONFIG` |

Além das métricas agregadas por fold, a seção agora também registra:

- previsões detalhadas por `fold x modelo x horizonte`
- resumo agregado por `modelo x h`
- curva de erro médio por horizonte na CV

O objetivo aqui não é competir com o holdout principal, mas avaliar estabilidade e generalização ao longo do tempo.

### Seção 9 - Dashboards e Análises Complementares
- Tabela de **custo-benefício**
- Gráficos de tempo total, tempo empilhado e tempo médio por abordagem
- Dashboard final para leitura executiva dos resultados

---

## Dados

### Contrato esperado

| Atributo | Esperado |
|----------|----------|
| Tipo de série | Univariada |
| Frequência | Mensal |
| Coluna temporal | `data` |
| Coluna alvo | `valores` |
| Ordenação | Cronológica crescente |
| Granularidade | 1 observação por mês |
| Horizonte padrão do holdout | 12 meses |
| Sazonalidade assumida | Anual (`S=12`) |

### Base incluída no repositório

| Atributo | Valor |
|----------|-------|
| Arquivo | `data/Base_DBP.xlsx` |
| Colunas | `data` (datetime), `valores` (arrecadação mensal) |
| Período | Agosto/2003 -> Agosto/2024 |
| Observações | 253 meses |

---

## Arquitetura do Repositório

O projeto está concentrado em dois artefatos principais:

| Arquivo | Função no experimento |
|---------|------------------------|
| `forecasting_experiments.ipynb` | executa o protocolo experimental completo |
| `viz.py` | centraliza a camada de visualização, comparação gráfica e export de figuras |

### Função do `viz.py`

O arquivo `viz.py` não treina modelos nem define o protocolo estatístico principal. Sua função é dar suporte a uma parte importante do experimento: a **leitura analítica dos resultados**.

Na prática, ele concentra rotinas reutilizáveis para:

- padronizar a geração de figuras
- evitar duplicação de código gráfico no notebook
- salvar painéis comparativos em `outputs/figures`
- manter consistência visual entre gráficos de EDA, forecasts, métricas, CV e custo-benefício

Isso ajuda a separar duas responsabilidades:

- o notebook fica responsável pela **lógica experimental**
- o `viz.py` fica responsável pela **camada de comunicação visual dos resultados**

Essa separação melhora manutenção, reprodutibilidade e legibilidade do experimento.

---

## Variáveis Globais

Essas configurações aparecem logo no início do notebook:

| Variável | Valor padrão | Descrição |
|----------|--------------|-----------|
| `SEED` | `42` | Semente global de reprodutibilidade |
| `TEST_HORIZON` | `12` | Horizonte do holdout principal |
| `SEASONAL_PERIOD` | `12` | Periodicidade sazonal mensal |
| `DC_WINDOW` | `12` | Janela de lags do pipeline Data-Centric |
| `ROOT` | `Path('.')` | Raiz do projeto |
| `DATA_PATH` | `ROOT / 'data'` | Pasta dos dados |
| `OUTPUT_PATH` | `ROOT / 'outputs'` | Pasta de saída para CSV e figuras |
| `EXP_NAME` | `'baseline_run_v1'` | Prefixo dos artefatos gerados |
| `FILE_NAME` | `'Base_DBP.xlsx'` | Arquivo de entrada |
| `DATE_COL` | `'data'` | Coluna de datas |
| `VALUE_COL` | `'valores'` | Coluna de valores |

---

## Artefatos de Saída

O repositório já contém artefatos de uma execução do experimento em `outputs/`.

### CSV / JSON

Arquivos consolidados em `outputs/csv/base_dbp/`:

| Arquivo | Conteúdo |
|---------|----------|
| `baseline_run_v1_eval_holdout.csv` | métricas do holdout principal |
| `baseline_run_v1_eval_approach_mean.csv` | média das métricas por abordagem |
| `baseline_run_v1_eval_best_models.csv` | melhor modelo por métrica |
| `baseline_run_v1_eval_horizon_bands.csv` | métricas por banda do horizonte |
| `baseline_run_v1_eval_all_preds.csv` | previsões consolidadas por data |
| `baseline_run_v1_eval_metadata.json` | metadados do experimento |
| `baseline_run_v1_cost_benefit.csv` | tabela de custo-benefício |
| `baseline_run_v1_summary_cb.csv` | resumo executivo de custo-benefício |

Arquivos por modelo:

| Padrão | Conteúdo |
|--------|----------|
| `*_preds.csv` | previsões do modelo no holdout |
| `*_artifacts.json` | métricas e parâmetros efetivos do modelo |

### Figuras

Exemplos de figuras já geradas em `outputs/figures/base_dbp/`:

| Arquivo | Conteúdo |
|---------|----------|
| `01_series_split.png` | split treino/teste da série |
| `02_stl_decomposition.png` | decomposição STL |
| `03_dc_forecasts.png` | previsões Data-Centric |
| `04_mc_forecasts.png` | previsões Model-Centric |
| `05_baseline_forecasts.png` | previsões dos baselines |
| `06_all_forecasts.png` | painel comparativo geral |
| `06b_ranking_rmse.png` | ranking por RMSE |
| `06c_ranking_smape.png` | ranking por sMAPE |
| `06d_approach_comparison.png` | comparação por abordagem |
| `06e_metrics_heatmap.png` | heatmap de métricas |
| `07_horizon_bands_rmse.png` | análise por bandas de horizonte |
| `07b_horizon_bands_smape.png` | bandas por sMAPE |
| `07c_all_forecasts_grid.png` | grade com todas as previsões |
| `08_cv_fold_rmse.png` | RMSE por fold na CV temporal |
| `08b_cv_fold_mase.png` | MASE por fold |
| `08c_cv_boxplot_rmse.png` | boxplot de RMSE na CV |
| `baseline_run_v1_cv_horizon_preds.csv` | previsões detalhadas por fold e horizonte na CV |
| `baseline_run_v1_cv_horizon_summary.csv` | métricas agregadas por modelo x horizonte na CV |
| `08d_cv_horizon_rmse.png` | RMSE médio por horizonte na CV |
| `baseline_run_v1_pareto_cost_accuracy.png` | Pareto custo vs. acurácia |
| `baseline_run_v1_tempo_total_por_modelo.png` | tempo total por modelo |
| `baseline_run_v1_tempo_empilhado_por_modelo.png` | decomposição do tempo por modelo |
| `baseline_run_v1_tempo_medio_por_abordagem.png` | tempo médio por abordagem |
| `baseline_run_v1_tempo_medio_por_etapa_abordagem.png` | tempo médio por etapa e abordagem |

---

## Requisitos

Pelas importações e pelo fluxo do notebook, o projeto depende de bibliotecas do ecossistema científico Python e de forecasting:

```bash
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
prophet
openpyxl
```

Instalação rápida:

```bash
pip install pandas numpy matplotlib scipy scikit-learn statsmodels xgboost optuna tensorflow prophet openpyxl
```

---

## Como Executar

1. Garanta que exista uma base **univariada e mensal** em `data/`.
2. Abra o notebook `forecasting_experiments.ipynb`.
3. Ajuste, se necessário, `FILE_NAME`, `DATE_COL` e `VALUE_COL` na configuração inicial.
4. Execute as células em ordem, preferencialmente com *Restart & Run All*.

Comando para iniciar:

```bash
jupyter notebook forecasting_experiments.ipynb
```

ou:

```bash
jupyter lab
```

A execução principal do experimento depende sobretudo das seções:

1. `1` e `2` para preparar ambiente e dados
2. `4`, `5` e `6` para gerar previsões por abordagem
3. `7` para consolidar métricas e salvar artefatos
4. `8` para rodar a validação cruzada temporal
5. `9` para gerar dashboards de custo-benefício e tempo

Se você trocar a base, preserve o contrato do experimento:

- uma série alvo por vez
- indexação mensal consistente
- dados ordenados cronologicamente
- quantidade mínima de histórico suficiente para lags sazonais e holdout

---

## CV Temporal no Model-Centric

Na validação cruzada temporal, o pipeline `Model-Centric` admite dois modos em `CV_CONFIG['mc_mode']`:

| Modo | Papel no experimento |
|------|----------------------|
| `nested` | reexecuta a seleção de hiperparâmetros dentro de cada fold |
| `pragmatic` | usa hiperparâmetros fixos em todos os folds |

Essa distinção é importante porque os dois modos respondem a perguntas diferentes.

### `nested`

Use quando a prioridade for **rigor metodológico**.

Em cada fold:

- o tuning acontece usando apenas o treino daquele fold
- o teste do fold permanece isolado
- a estimativa final de CV fica mais próxima de uma medida limpa de generalização

Esse modo é o mais apropriado para responder:

> se eu repetir todo o processo de seleção e treino ao longo do tempo, qual desempenho devo esperar?

O custo é maior, porque SARIMA, XGBoost e LSTM precisam repetir busca e ajuste em cada fold.

### `pragmatic`

Use quando a prioridade for **viabilidade computacional** e **estabilidade de uma configuração fixa**.

Nesse modo:

- os hiperparâmetros são mantidos constantes em todos os folds
- a CV fica mais barata e mais rápida
- o foco passa a ser a robustez operacional do pipeline, não a avaliação limpa do processo de tuning

Esse modo responde melhor a perguntas como:

> com uma configuração já escolhida, o pipeline se mantém estável em diferentes recortes temporais?

### Por que não usar apenas um modo?

- Usar só `nested` aumenta muito o custo computacional do experimento
- Usar só `pragmatic` enfraquece a validade metodológica da comparação, se os parâmetros fixos vierem de ajuste anterior

Manter os dois modos é útil porque:

- `nested` cobre a **validade científica** da comparação
- `pragmatic` cobre a **viabilidade prática** e a estabilidade operacional

Isso é especialmente relevante aqui porque o `Data-Centric` quase não depende de tuning, enquanto o `Model-Centric` depende fortemente dele. Sem essa separação, a comparação entre abordagens pode ficar enviesada ou pouco clara.

---

## Estrutura do Repositório

```text
.
|-- forecasting_experiments.ipynb
|-- viz.py
|-- README.md
|-- data/
`-- outputs/
```

## Observação Final

Este `README` documenta o racional e a estrutura do experimento, não um conjunto definitivo de conclusões. Os resultados consolidados ainda dependem das execuções nas diferentes bases que serão analisadas. Por isso, o foco do texto está em explicitar:

- o contrato de entrada
- a lógica de comparação entre abordagens
- o papel das métricas, da CV temporal e dos baselines
- a organização técnica necessária para reprodução

Com isso, o usuário consegue entender o que o experimento pretende medir antes mesmo de inspecionar os resultados finais.
