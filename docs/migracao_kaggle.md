# Migração dos experimentos para Kaggle

O armazenamento principal do protocolo é o Google Drive. O backup do notebook cria um dataset privado no Kaggle chamado, por padrão, `plantdoc-yolov8-tcc-checkpoints`. Ele contém `runs_staged.zip`, com os resumos, métricas e checkpoints `best.pt` e `last.pt`.

## Autenticação segura

Gere um token em **Kaggle > Settings > API**. No Colab, adicione o valor ao painel **Secrets** com o nome `KAGGLE_API_TOKEN` e habilite o acesso ao notebook. Não cole o token em uma célula, arquivo ou commit.

O Kaggle CLI atual aceita o token pela variável de ambiente `KAGGLE_API_TOKEN`. O notebook cria o dataset como privado e usa `kaggle datasets version` nos envios seguintes.

## Restaurar no Kaggle

Crie um Kaggle Notebook com GPU e Internet habilitadas. Em uma célula, substitua `SEU_USUARIO` e execute:

```bash
git clone https://github.com/SoulStorm0/plantdoc-yolov8-tcc.git /kaggle/working/plantdoc-yolov8-tcc
pip install -q -e /kaggle/working/plantdoc-yolov8-tcc
kaggle datasets download SEU_USUARIO/plantdoc-yolov8-tcc-checkpoints -p /kaggle/working/restore --unzip
unzip -o /kaggle/working/restore/runs_staged.zip -d /kaggle/working/TCC_PlantDoc
```

Prepare novamente o PlantDoc a partir da fonte oficial para manter o mesmo split determinístico:

```bash
cd /kaggle/working/plantdoc-yolov8-tcc
git clone --depth 1 --no-checkout https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset.git /kaggle/working/plantdoc_official
python scripts/prepare_official_plantdoc.py --repo /kaggle/working/plantdoc_official --output /kaggle/working/plantdoc_yolo_27 --min-class-instances 20
```

Continue um único item pendente. Este exemplo retoma o índice 4 da busca:

```bash
python -m plantdoc_tcc staged \
  --data /kaggle/working/plantdoc_yolo_27/data.yaml \
  --config configs/colab_protocol.json \
  --project /kaggle/working/TCC_PlantDoc/runs_staged \
  --phase search --run-index 4 --device 0
```

Consulte os próximos índices com:

```bash
python -m plantdoc_tcc status \
  --config configs/colab_protocol.json \
  --project /kaggle/working/TCC_PlantDoc/runs_staged
```

Antes de encerrar a sessão do Kaggle, gere uma nova versão do backup. Reaproveite a célula de backup do notebook do Colab, alterando `RUN_ROOT` para `/kaggle/working/TCC_PlantDoc/runs_staged`, ou compacte a pasta e execute `kaggle datasets version`.

## Cuidados

- O diretório `/kaggle/input` é somente leitura; restaure sempre em `/kaggle/working`.
- Nunca treine com caminhos diferentes de classes ou com outro split do PlantDoc.
- Não execute o teste final durante a migração. Ele permanece reservado para depois da seleção completa.
- Confirme que `protocol_summary.json` e o `last.pt` mais recente estão presentes antes de continuar.
