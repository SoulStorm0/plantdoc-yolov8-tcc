# TCC - Detecção de doenças foliares com YOLOv8

Pipeline reproduzível para os experimentos propostos no TCC: PlantDoc em 640x640, transfer learning com YOLOv8, comparação entre loss padrão, ponderação por classe e Focal Loss, busca de hiperparâmetros e avaliação externa estratificada por iluminação.

## O que este repositório resolve

- audita a distribuição de imagens e instâncias por classe antes do treino;
- calcula pesos apenas no conjunto de treino (sem vazamento do teste);
- compara `baseline`, `class_weighted` e `focal` com a mesma semente e partição;
- pesquisa `lr0` entre `1e-3` e `1e-5`, momentum, weight decay, warmup e cosine annealing;
- registra precisão, revocação, F1, AP50 e AP50-95 global e por classe;
- avalia imagens externas somente quando há bounding boxes revisadas e separa os resultados por condição de iluminação.

> O total de 2.598 imagens citado no texto é do PlantDoc como um todo. A edição de detecção publicada em fontes distintas pode ter outra quantidade e até 29/30 categorias. O comando de auditoria interrompe o experimento se o `data.yaml` não tiver exatamente o número esperado de classes.

## Execução rápida no Google Colab

Abra [`notebooks/plantdoc_yolov8_colab.ipynb`](notebooks/plantdoc_yolov8_colab.ipynb), selecione uma GPU e execute as células. O notebook instala o projeto, recebe o ZIP exportado do Roboflow em formato **YOLOv8**, audita os rótulos e inicia o grid.

No terminal:

```bash
python -m pip install -e ".[dev]"
python -m plantdoc_tcc audit --data /caminho/plantdoc/data.yaml --expected-classes 27
python -m plantdoc_tcc train --data /caminho/plantdoc/data.yaml --config configs/experiments.json --epoch 100
python -m plantdoc_tcc evaluate --weights runs/plantdoc/<run>/weights/best.pt --data /caminho/plantdoc/data.yaml --split test
```

O grid completo tem 108 execuções (3 épocas x 2 batches x 3 learning rates x 2 momentums x 3 weight decays) para cada estratégia de loss e é caro. Recomenda-se usar `--epoch 100` para a triagem e promover apenas as melhores combinações para 200/300 épocas. Para validar a instalação sem GPU:

```bash
python -m unittest discover -s tests -v
python -m plantdoc_tcc plan --config configs/experiments.json
```

Para um teste de integração de uma época com dados sintéticos (não produz resultado científico):

```bash
python scripts/create_smoke_dataset.py
python -m plantdoc_tcc train --data datasets/smoke/data.yaml --config configs/smoke.json --device cpu
```

## Dados

Exporte o PlantDoc do Roboflow no formato YOLOv8, mantendo `train`, `valid` e `test`. O particionamento deve ser congelado e registrado em `data.yaml`. Não versione imagens, pesos ou credenciais.

Estrutura esperada:

```text
plantdoc/
  data.yaml
  train/images  train/labels
  valid/images  valid/labels
  test/images   test/labels
```

O projeto não redistribui o dataset. A fonte acadêmica é Singh et al. (2020), e o dataset de detecção deve ser obtido da publicação original ou do projeto público no Roboflow, respeitando a licença CC BY 4.0.

## Protocolo das imagens externas

Veja [`docs/protocolo_avaliacao_externa.md`](docs/protocolo_avaliacao_externa.md). Cada imagem deve ter anotação YOLO revisada por duas pessoas e uma linha em `metadata.csv`. Sem ground truth, o pipeline permite inferência qualitativa, mas não calcula mAP ou F1.

## Reprodutibilidade

- Python 3.10-3.12 e dependências fixadas em `pyproject.toml`;
- seed 42, operações determinísticas quando suportadas e AMP desligável;
- resultados em `runs/plantdoc`, configurações e versão das bibliotecas salvas pelo Ultralytics;
- o conjunto de teste não participa da seleção de hiperparâmetros;
- a escolha final usa validação; o teste é executado uma única vez para estimativa imparcial.

## Observação metodológica

YOLOv8 já usa *Distribution Focal Loss* para regressão das caixas; isso não equivale à Focal Loss de classificação sugerida na avaliação. A implementação deste repositório substitui somente o BCE da classificação. A ponderação usa `pos_weight` por classe, conforme o mecanismo documentado pelo Ultralytics, com pesos suavizados e limitados para evitar gradientes extremos.
