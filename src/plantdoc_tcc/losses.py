from __future__ import annotations

from typing import Sequence


def focal_bce_elementwise(logits, targets, gamma: float = 2.0, alpha: float = 0.25, pos_weight=None):
    """BCE focal sem redução, compatível com a classificação do v8DetectionLoss."""
    import torch.nn.functional as F

    base = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
    probabilities = logits.sigmoid()
    pt = targets * probabilities + (1.0 - targets) * (1.0 - probabilities)
    alpha_factor = targets * alpha + (1.0 - targets) * (1.0 - alpha)
    return base * ((1.0 - pt) ** gamma) * alpha_factor


try:
    import torch
    from torch import nn
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.loss import v8DetectionLoss

    class ClassificationLoss(nn.Module):
        """BCE ponderada ou focal sem redução, como esperado pelo v8DetectionLoss."""

        def __init__(self, class_weights, gamma, alpha, device):
            super().__init__()
            value = class_weights.to(device) if class_weights is not None else None
            self.register_buffer("class_weights", value)
            self.gamma = gamma
            self.alpha = alpha

        def forward(self, pred, target):
            if self.gamma is None:
                return nn.functional.binary_cross_entropy_with_logits(
                    pred, target, reduction="none", pos_weight=self.class_weights
                )
            return focal_bce_elementwise(
                pred, target, gamma=self.gamma, alpha=self.alpha, pos_weight=self.class_weights
            )


    class BalancedDetectionLoss(v8DetectionLoss):
        """Loss YOLOv8 que altera somente o termo de classificação."""

        def __init__(self, model, tal_topk=10, tal_topk2=None):
            super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
            weights = None
            if model.loss_class_weights is not None:
                weights = torch.tensor(model.loss_class_weights, dtype=torch.float32)
            self.bce = ClassificationLoss(
                weights, model.loss_focal_gamma, model.loss_focal_alpha, self.device
            )


    class BalancedDetectionModel(DetectionModel):
        """Classe em nível de módulo para permitir serialização segura do checkpoint."""

        loss_class_weights = None
        loss_focal_gamma = None
        loss_focal_alpha = 0.25

        def init_criterion(self):
            return BalancedDetectionLoss(self)

except ImportError:  # Permite auditoria e testes de dados sem instalar o stack de treino.
    ClassificationLoss = None
    BalancedDetectionLoss = None
    BalancedDetectionModel = None


def make_trainer_class(
    weights: Sequence[float] | None = None,
    focal_gamma: float | None = None,
    focal_alpha: float = 0.25,
):
    """Cria trainer com loss configurada; imports tardios mantêm auditoria/testes leves."""
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils import RANK

    if BalancedDetectionModel is None:
        raise RuntimeError("PyTorch e Ultralytics são necessários para criar o trainer.")
    BalancedDetectionModel.loss_class_weights = None if weights is None else list(weights)
    BalancedDetectionModel.loss_focal_gamma = focal_gamma
    BalancedDetectionModel.loss_focal_alpha = focal_alpha

    class BalancedTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = BalancedDetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
            if weights:
                model.load(weights)
            return model

    return BalancedTrainer
