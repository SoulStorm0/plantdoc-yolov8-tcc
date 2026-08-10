# Protocolo de avaliação externa

## Objetivo e separação

O conjunto externo mede *domain shift* e não pode ser usado para treino, ajuste de limiar ou escolha de hiperparâmetros. Antes da coleta, defina culturas/classes-alvo e procure obter pelo menos 30 instâncias por classe e por estrato de iluminação. Quando isso não for possível, reporte intervalo de confiança e a amostra exata, sem generalizar o resultado.

## Captura

Registre imagens de diferentes aparelhos, propriedades, fundos, distâncias e ângulos. Não capture rajadas quase idênticas da mesma folha; imagens da mesma planta/sessão recebem um `group_id` comum. Remova rostos, placas, geolocalização EXIF e outros dados pessoais antes do armazenamento.

Estratos obrigatórios de iluminação:

- `baixa`: sombra intensa ou baixa luminosidade;
- `difusa`: céu nublado/sombra uniforme;
- `direta`: sol direto sem contraluz dominante;
- `contraluz`: fonte de luz atrás da folha.

## Ground truth

1. Um especialista ou estudante treinado delimita todas as folhas/lesões conforme o guia de classes do PlantDoc.
2. Um segundo anotador revisa a classe, a cobertura da caixa e objetos omitidos, sem ver a previsão do modelo.
3. Divergências são adjudicadas por especialista. Registre os dois responsáveis e `adjudicated=true`.
4. Uma segunda rede pode sugerir caixas para acelerar o trabalho, mas nunca é aceita como verdade-terreno sem revisão humana.
5. O conjunto fica congelado antes da primeira avaliação do modelo.

Formato de `metadata.csv`:

```csv
image,lighting,device,site,group_id,annotator_1,annotator_2,adjudicated
images/IMG_0001.jpg,direta,Galaxy_A54,local_01,planta_004,Ana,Bruno,true
```

As caixas ficam em `labels/<mesmo_nome>.txt`, no formato YOLO: `classe x_centro y_centro largura altura`, coordenadas normalizadas.

## Relato

Calcule precisão, revocação, F1, mAP@50 e mAP@50:95 global, por classe e por iluminação. Compare cada estrato ao teste interno e reporte a diferença absoluta em pontos percentuais. Apresente também número de imagens/instâncias, IC bootstrap de 95%, matriz de confusão e exemplos de falso positivo/negativo. Imagens sem ground truth servem apenas para análise qualitativa.

