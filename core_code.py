"""Core AAMRBP modules corresponding to the manuscript method section.

The file intentionally omits datasets, Qwen3-VL loading, LoRA training,
distributed launch code, checkpoint management, and production inference.
It implements only the method's central computation:

1. Project Qwen3-VL visual tokens into a 512-D numerical memory.
2. Use Stage-I boxes to attend to target-region visual tokens, then fuse the
   regional evidence and spatial prior into object queries.
3. Decode target-level and decoration-level evidence with two decoders.
4. Predict presence, geometry, dimensions, decoration count, and >=9 class.
5. Perform one-to-one Hungarian matching and compute the multi-task loss.

Stage I is represented by ``target_prior_boxes``.  In the complete system,
these normalized boxes are produced by the localization LoRA before being
passed to this Stage-II core.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TargetBatch(TypedDict, total=False):
    """Ground truth for one image.

    All boxes use normalized ``[cx, cy, width, height]`` coordinates.
    The manuscript dataset contains one target per Stage-II image, but the
    representation also supports multiple targets.
    """

    boxes: Tensor
    decoration_counts: Tensor
    ge9_labels: Tensor
    length_width_um: Tensor


@dataclass(frozen=True)
class AAMRBPConfig:
    """Architecture values used by the paper implementation."""

    visual_dim: int = 2560
    decoder_dim: int = 512
    decoder_layers: int = 3
    attention_heads: int = 8
    num_object_queries: int = 8
    num_decoration_queries: int = 24
    max_decoration_count: float = 20.0
    ge9_threshold: int = 9
    presence_positive_weight: float = 5.0
    dropout: float = 0.1


@dataclass(frozen=True)
class LossWeights:
    """Weights matching the checked-in numerical-head implementation."""

    presence: float = 1.0
    bbox_l1: float = 5.0
    bbox_giou: float = 2.0
    morphometry: float = 0.05
    ge9: float = 2.0
    target_count: float = 1.0
    auxiliary_count: float = 0.5


@dataclass(frozen=True)
class MatchWeights:
    """Weights used to build the Hungarian assignment cost."""

    bbox_l1: float = 5.0
    bbox_giou: float = 2.0
    ge9: float = 1.0
    count: float = 1.0


class MLP(nn.Module):
    """Small prediction/encoding network used throughout AAMRBP."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        layers: int,
    ) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        for layer_index in range(layers):
            in_features = input_dim if layer_index == 0 else hidden_dim
            out_features = output_dim if layer_index == layers - 1 else hidden_dim
            modules.append(nn.Linear(in_features, out_features))
            if layer_index < layers - 1:
                modules.append(nn.GELU())
        self.network = nn.Sequential(*modules)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs)


def cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    """Convert normalized center boxes to corner boxes."""

    center_x, center_y, width, height = boxes.unbind(dim=-1)
    return torch.stack(
        (
            center_x - width / 2.0,
            center_y - height / 2.0,
            center_x + width / 2.0,
            center_y + height / 2.0,
        ),
        dim=-1,
    )


def box_area(boxes: Tensor) -> Tensor:
    sizes = (boxes[..., 2:] - boxes[..., :2]).clamp(min=0.0)
    return sizes[..., 0] * sizes[..., 1]


def pairwise_generalized_iou(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    """Pairwise generalized IoU for ``xyxy`` boxes."""

    top_left = torch.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = torch.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    intersection_size = (bottom_right - top_left).clamp(min=0.0)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]

    area_a = box_area(boxes_a)[:, None]
    area_b = box_area(boxes_b)[None, :]
    union = (area_a + area_b - intersection).clamp(min=1e-7)
    iou = intersection / union

    enclosing_top_left = torch.minimum(
        boxes_a[:, None, :2], boxes_b[None, :, :2]
    )
    enclosing_bottom_right = torch.maximum(
        boxes_a[:, None, 2:], boxes_b[None, :, 2:]
    )
    enclosing_size = (enclosing_bottom_right - enclosing_top_left).clamp(min=0.0)
    enclosing_area = (
        enclosing_size[..., 0] * enclosing_size[..., 1]
    ).clamp(min=1e-7)
    return iou - (enclosing_area - union) / enclosing_area


def exact_linear_assignment(cost: Tensor) -> tuple[Tensor, Tensor]:
    """Exact one-to-one assignment for the paper's eight object queries.

    The original project uses SciPy's Hungarian solver.  Stage-II images in
    the manuscript contain one annotated target, so an exact permutation
    search over at most eight queries is compact and keeps this core file
    dependency-free.  The returned tensors index prediction and target rows.
    """

    num_queries, num_targets = cost.shape
    if num_targets == 0:
        empty = torch.empty(0, dtype=torch.long, device=cost.device)
        return empty, empty
    if num_targets > num_queries:
        raise ValueError(
            f"Cannot match {num_targets} targets to {num_queries} object queries"
        )

    detached = cost.detach().float().cpu()
    target_indices = tuple(range(num_targets))
    best_queries: tuple[int, ...] | None = None
    best_cost = float("inf")
    for query_indices in itertools.permutations(range(num_queries), num_targets):
        candidate = sum(
            float(detached[query_index, target_index])
            for query_index, target_index in zip(query_indices, target_indices)
        )
        if candidate < best_cost:
            best_cost = candidate
            best_queries = query_indices
    assert best_queries is not None
    return (
        torch.tensor(best_queries, dtype=torch.long, device=cost.device),
        torch.arange(num_targets, dtype=torch.long, device=cost.device),
    )


def normalize_image_metadata(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    image_size_px: Tensor | None,
    um_per_pixel: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """Normalize image dimensions and per-image calibration to ``[B, 2]``."""

    if image_size_px is None:
        image_size_px = torch.full(
            (batch_size, 2), 1024.0, device=device, dtype=dtype
        )
    else:
        image_size_px = image_size_px.to(device=device, dtype=dtype)
        if image_size_px.ndim == 1:
            image_size_px = image_size_px.unsqueeze(0).expand(batch_size, -1)
    if image_size_px.shape != (batch_size, 2):
        raise ValueError("image_size_px must have shape [B, 2] as [width, height]")

    if um_per_pixel is None:
        um_per_pixel = torch.full(
            (batch_size, 2), 0.125, device=device, dtype=dtype
        )
    else:
        um_per_pixel = um_per_pixel.to(device=device, dtype=dtype)
        if um_per_pixel.ndim == 0:
            um_per_pixel = um_per_pixel.repeat(batch_size, 2)
        elif um_per_pixel.ndim == 1:
            if um_per_pixel.numel() == batch_size:
                um_per_pixel = um_per_pixel[:, None].expand(-1, 2)
            elif um_per_pixel.numel() == 2:
                um_per_pixel = um_per_pixel[None, :].expand(batch_size, -1)
    if um_per_pixel.shape != (batch_size, 2):
        raise ValueError(
            "um_per_pixel must be scalar, [B], [2], or [B, 2] for x/y calibration"
        )
    return image_size_px, um_per_pixel


def calibrated_length_width(
    boxes_cxcywh: Tensor,
    image_size_px: Tensor,
    um_per_pixel: Tensor,
) -> Tensor:
    """Convert normalized box geometry to calibrated ``[length, width]`` μm."""

    box_width_px = boxes_cxcywh[..., 2] * image_size_px[:, None, 0]
    box_height_px = boxes_cxcywh[..., 3] * image_size_px[:, None, 1]
    box_width_um = box_width_px * um_per_pixel[:, None, 0]
    box_height_um = box_height_px * um_per_pixel[:, None, 1]
    length_um = torch.maximum(box_width_um, box_height_um)
    width_um = torch.minimum(box_width_um, box_height_um)
    return torch.stack((length_um, width_um), dim=-1)


class TargetPerceptionEnhancer(nn.Module):
    """Turn Stage-I boxes into spatially grounded target-aware queries.

    The box coordinates provide a geometric prior.  A differentiable soft box
    mask then pools the visual tokens inside that region.  A learned gate fuses
    the geometric and regional features before they are added to selected
    object queries.  Unguided queries remain available for global evidence.
    """

    def __init__(self, config: AAMRBPConfig) -> None:
        super().__init__()
        self.num_object_queries = config.num_object_queries
        self.decoder_dim = config.decoder_dim
        self.box_encoder = MLP(4, config.decoder_dim, config.decoder_dim, 3)
        self.region_encoder = MLP(
            config.decoder_dim, config.decoder_dim, config.decoder_dim, 2
        )
        self.fusion_gate = MLP(
            config.decoder_dim * 2,
            config.decoder_dim,
            config.decoder_dim,
            2,
        )
        self.output_norm = nn.LayerNorm(config.decoder_dim)
        self.spatial_sharpness = nn.Parameter(torch.tensor(12.0))

    @staticmethod
    def _make_token_coordinates(
        memory: Tensor,
        visual_token_coords: Tensor | None,
    ) -> Tensor:
        batch_size, token_count, _ = memory.shape
        if visual_token_coords is None:
            grid_side = int(math.isqrt(token_count))
            if grid_side * grid_side != token_count:
                raise ValueError(
                    "visual_token_coords is required when visual tokens do not "
                    "form a square grid"
                )
            axis = (
                torch.arange(grid_side, device=memory.device, dtype=memory.dtype)
                + 0.5
            ) / grid_side
            grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
            coordinates = torch.stack((grid_x, grid_y), dim=-1).reshape(
                1, token_count, 2
            )
            return coordinates.expand(batch_size, -1, -1)

        coordinates = visual_token_coords.to(device=memory.device, dtype=memory.dtype)
        if coordinates.ndim == 2:
            coordinates = coordinates.unsqueeze(0).expand(batch_size, -1, -1)
        if coordinates.shape != (batch_size, token_count, 2):
            raise ValueError("visual_token_coords must have shape [N, 2] or [B, N, 2]")
        return coordinates

    def _spatial_attention(
        self,
        memory: Tensor,
        target_prior_boxes: Tensor,
        target_prior_mask: Tensor,
        visual_padding_mask: Tensor,
        visual_token_coords: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        coordinates = self._make_token_coordinates(memory, visual_token_coords)
        centers = target_prior_boxes[..., :2].to(memory).unsqueeze(2)
        half_sizes = (
            target_prior_boxes[..., 2:].to(memory).clamp(min=1e-4).unsqueeze(2)
            / 2.0
        )
        normalized_distance = (
            (coordinates.unsqueeze(1) - centers).abs() / half_sizes
        )
        sharpness = self.spatial_sharpness.abs().clamp(min=1.0)
        attention = torch.sigmoid(
            sharpness * (1.0 - normalized_distance[..., 0])
        ) * torch.sigmoid(sharpness * (1.0 - normalized_distance[..., 1]))
        valid = (
            target_prior_mask.unsqueeze(-1)
            & (~visual_padding_mask).unsqueeze(1)
        )
        attention = attention * valid.to(attention.dtype)
        normalized_attention = attention / attention.sum(dim=-1, keepdim=True).clamp(
            min=1e-6
        )
        regional_features = torch.einsum(
            "bpn,bnd->bpd", normalized_attention, memory
        )
        return attention, regional_features

    def forward(
        self,
        object_queries: Tensor,
        memory: Tensor,
        target_prior_boxes: Tensor,
        target_prior_mask: Tensor,
        visual_padding_mask: Tensor,
        visual_token_coords: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size, num_priors, _ = target_prior_boxes.shape
        spatial_attention, regional_features = self._spatial_attention(
            memory,
            target_prior_boxes,
            target_prior_mask,
            visual_padding_mask,
            visual_token_coords,
        )
        box_features = self.box_encoder(target_prior_boxes.float())
        regional_features = self.region_encoder(regional_features)
        gate = self.fusion_gate(
            torch.cat((box_features, regional_features), dim=-1)
        ).sigmoid()
        enhanced_priors = self.output_norm(
            box_features + gate * regional_features
        )
        enhanced_priors = enhanced_priors * target_prior_mask.unsqueeze(-1).float()

        # Some queries receive the localization prior; the remaining queries
        # stay purely learnable and can retrieve complementary global evidence.
        guided_count = min(num_priors, self.num_object_queries)
        prior_slots = object_queries.new_zeros(
            batch_size, self.num_object_queries, object_queries.shape[-1]
        )
        prior_slots[:, :guided_count] = enhanced_priors[:, :guided_count]
        return object_queries + prior_slots, spatial_attention, enhanced_priors


class AAMRBPCore(nn.Module):
    """Target-prior-enhanced dual-decoder numerical core."""

    def __init__(self, config: AAMRBPConfig = AAMRBPConfig()) -> None:
        super().__init__()
        self.config = config

        # Equation: M = LayerNorm(W_v H_v + b_v)
        self.visual_projection = nn.Sequential(
            nn.Linear(config.visual_dim, config.decoder_dim),
            nn.LayerNorm(config.decoder_dim),
        )

        self.object_queries = nn.Embedding(
            config.num_object_queries, config.decoder_dim
        )
        self.decoration_queries = nn.Embedding(
            config.num_decoration_queries, config.decoder_dim
        )
        self.target_perception = TargetPerceptionEnhancer(config)

        object_layer = nn.TransformerDecoderLayer(
            d_model=config.decoder_dim,
            nhead=config.attention_heads,
            dim_feedforward=config.decoder_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        decoration_layer = nn.TransformerDecoderLayer(
            d_model=config.decoder_dim,
            nhead=config.attention_heads,
            dim_feedforward=config.decoder_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.object_decoder = nn.TransformerDecoder(
            object_layer, num_layers=config.decoder_layers
        )
        self.decoration_decoder = nn.TransformerDecoder(
            decoration_layer, num_layers=config.decoder_layers
        )

        self.presence_head = MLP(config.decoder_dim, config.decoder_dim, 1, 2)
        self.bbox_head = MLP(config.decoder_dim, config.decoder_dim, 4, 3)
        self.ge9_head = MLP(config.decoder_dim, config.decoder_dim, 1, 2)
        self.count_head = MLP(config.decoder_dim, config.decoder_dim, 1, 2)
        self.decoration_presence_head = MLP(
            config.decoder_dim, config.decoder_dim, 1, 2
        )

    def forward(
        self,
        visual_tokens: Tensor,
        visual_padding_mask: Tensor,
        target_prior_boxes: Tensor,
        target_prior_mask: Tensor,
        visual_token_coords: Tensor | None = None,
        image_size_px: Tensor | None = None,
        um_per_pixel: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Run the paper's Stage-II numerical decoding.

        Args:
            visual_tokens: Qwen3-VL image-token states ``[B, N, D_visual]``.
            visual_padding_mask: ``[B, N]``; ``True`` marks padded tokens.
            target_prior_boxes: Stage-I boxes ``[B, P, 4]`` in normalized
                ``cxcywh`` form.
            target_prior_mask: ``[B, P]`` validity mask.
            visual_token_coords: Optional normalized ``[B, N, 2]`` spatial
                coordinates for Qwen visual tokens. Square grids are inferred.
            image_size_px: Per-image ``[width, height]`` in pixels.
            um_per_pixel: Image-specific x/y scale calibration.
        """

        if visual_tokens.ndim != 3:
            raise ValueError("visual_tokens must have shape [B, N, D]")
        batch_size = visual_tokens.shape[0]
        memory = self.visual_projection(visual_tokens.float())

        base_object_queries = self.object_queries.weight.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        object_queries, target_attention, enhanced_target_priors = (
            self.target_perception(
                base_object_queries,
                memory,
                target_prior_boxes,
                target_prior_mask,
                visual_padding_mask,
                visual_token_coords,
            )
        )
        object_tokens = self.object_decoder(
            tgt=object_queries,
            memory=memory,
            memory_key_padding_mask=visual_padding_mask,
        )

        decoration_queries = self.decoration_queries.weight.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        decoration_tokens = self.decoration_decoder(
            tgt=decoration_queries,
            memory=memory,
            memory_key_padding_mask=visual_padding_mask,
        )

        presence_logits = self.presence_head(object_tokens).squeeze(-1)
        boxes_cxcywh = self.bbox_head(object_tokens).sigmoid()
        ge9_logits = self.ge9_head(object_tokens).squeeze(-1)
        decoration_counts = (
            self.count_head(object_tokens).squeeze(-1).sigmoid()
            * self.config.max_decoration_count
        )
        decoration_presence_logits = self.decoration_presence_head(
            decoration_tokens
        ).squeeze(-1)
        auxiliary_decoration_count = decoration_presence_logits.sigmoid().sum(dim=1)

        image_size_px, um_per_pixel = normalize_image_metadata(
            batch_size,
            visual_tokens.device,
            boxes_cxcywh.dtype,
            image_size_px,
            um_per_pixel,
        )
        length_width_um = calibrated_length_width(
            boxes_cxcywh, image_size_px, um_per_pixel
        )

        # The continuous count and >=9 branch remain independent.  In
        # particular, the rounded count is not forced to agree with the class.
        return {
            "presence_logits": presence_logits,
            "presence_probability": presence_logits.sigmoid(),
            "boxes_cxcywh": boxes_cxcywh,
            "boxes_xyxy": cxcywh_to_xyxy(boxes_cxcywh).clamp(0.0, 1.0),
            "length_width_um": length_width_um,
            "ge9_logits": ge9_logits,
            "ge9_probability": ge9_logits.sigmoid(),
            "ge9_class": (ge9_logits.sigmoid() >= 0.5).long(),
            "decoration_count": decoration_counts,
            "decoration_count_rounded": decoration_counts.round().long(),
            "decoration_presence_logits": decoration_presence_logits,
            "auxiliary_decoration_count": auxiliary_decoration_count,
            "object_tokens": object_tokens,
            "decoration_tokens": decoration_tokens,
            "target_attention": target_attention,
            "enhanced_target_priors": enhanced_target_priors,
        }


class AAMRBPLoss(nn.Module):
    """Hungarian matching plus the manuscript's multi-task objective."""

    def __init__(
        self,
        config: AAMRBPConfig = AAMRBPConfig(),
        loss_weights: LossWeights = LossWeights(),
        match_weights: MatchWeights = MatchWeights(),
    ) -> None:
        super().__init__()
        self.config = config
        self.loss_weights = loss_weights
        self.match_weights = match_weights

    def _match_one(
        self,
        prediction_boxes: Tensor,
        prediction_ge9_logits: Tensor,
        prediction_counts: Tensor,
        target: TargetBatch,
    ) -> tuple[Tensor, Tensor]:
        target_boxes = target["boxes"].to(prediction_boxes)
        target_ge9 = target["ge9_labels"].to(prediction_boxes)
        target_counts = target["decoration_counts"].to(prediction_boxes)

        bbox_l1_cost = torch.cdist(prediction_boxes, target_boxes, p=1)
        giou_cost = -pairwise_generalized_iou(
            cxcywh_to_xyxy(prediction_boxes), cxcywh_to_xyxy(target_boxes)
        )
        ge9_cost = torch.abs(
            prediction_ge9_logits.sigmoid()[:, None] - target_ge9[None, :]
        )
        count_cost = torch.abs(
            prediction_counts[:, None] - target_counts[None, :]
        ) / self.config.max_decoration_count
        total_cost = (
            self.match_weights.bbox_l1 * bbox_l1_cost
            + self.match_weights.bbox_giou * giou_cost
            + self.match_weights.ge9 * ge9_cost
            + self.match_weights.count * count_cost
        )
        return exact_linear_assignment(total_cost)

    def forward(
        self,
        outputs: dict[str, Tensor],
        targets: list[TargetBatch],
        image_size_px: Tensor | None = None,
        um_per_pixel: Tensor | None = None,
    ) -> dict[str, Tensor]:
        batch_size, num_queries = outputs["presence_logits"].shape
        if len(targets) != batch_size:
            raise ValueError("targets must contain one dictionary per image")

        presence_targets = torch.zeros_like(outputs["presence_logits"])
        matched_prediction_indices: list[Tensor] = []
        matched_batch_indices: list[Tensor] = []
        matched_target_indices: list[Tensor] = [
            torch.empty(
                0,
                dtype=torch.long,
                device=outputs["presence_logits"].device,
            )
            for _ in range(batch_size)
        ]

        for batch_index, target in enumerate(targets):
            num_targets = int(target["boxes"].shape[0])
            if num_targets == 0:
                continue
            prediction_indices, target_indices = self._match_one(
                outputs["boxes_cxcywh"][batch_index],
                outputs["ge9_logits"][batch_index],
                outputs["decoration_count"][batch_index],
                target,
            )
            presence_targets[batch_index, prediction_indices] = 1.0
            matched_batch_indices.append(
                torch.full_like(prediction_indices, batch_index)
            )
            matched_prediction_indices.append(prediction_indices)
            matched_target_indices[batch_index] = target_indices

        positive_weight = outputs["presence_logits"].new_tensor(
            self.config.presence_positive_weight
        )
        loss_presence = F.binary_cross_entropy_with_logits(
            outputs["presence_logits"],
            presence_targets,
            pos_weight=positive_weight,
        )

        zero = outputs["boxes_cxcywh"].sum() * 0.0
        loss_bbox_l1 = zero
        loss_bbox_giou = zero
        loss_morphometry = zero
        loss_ge9 = zero
        loss_target_count = zero

        if matched_prediction_indices:
            batch_indices = torch.cat(matched_batch_indices)
            prediction_indices = torch.cat(matched_prediction_indices)
            target_indices_by_image = matched_target_indices

            prediction_boxes = outputs["boxes_cxcywh"][
                batch_indices, prediction_indices
            ]
            prediction_ge9 = outputs["ge9_logits"][
                batch_indices, prediction_indices
            ]
            prediction_counts = outputs["decoration_count"][
                batch_indices, prediction_indices
            ]

            target_boxes = torch.cat(
                [
                    targets[batch]["boxes"][indices]
                    for batch, indices in enumerate(target_indices_by_image)
                    if indices.numel() > 0
                ]
            ).to(prediction_boxes)
            target_ge9 = torch.cat(
                [
                    targets[batch]["ge9_labels"][indices]
                    for batch, indices in enumerate(target_indices_by_image)
                    if indices.numel() > 0
                ]
            ).to(prediction_ge9)
            target_counts = torch.cat(
                [
                    targets[batch]["decoration_counts"][indices]
                    for batch, indices in enumerate(target_indices_by_image)
                    if indices.numel() > 0
                ]
            ).to(prediction_counts)

            loss_bbox_l1 = F.l1_loss(prediction_boxes, target_boxes)
            giou = pairwise_generalized_iou(
                cxcywh_to_xyxy(prediction_boxes), cxcywh_to_xyxy(target_boxes)
            )
            loss_bbox_giou = (1.0 - torch.diagonal(giou)).mean()
            loss_ge9 = F.binary_cross_entropy_with_logits(
                prediction_ge9, target_ge9
            )
            loss_target_count = F.smooth_l1_loss(
                prediction_counts, target_counts
            )

            if image_size_px is None:
                image_size_px = prediction_boxes.new_full((batch_size, 2), 1024.0)
            if um_per_pixel is None:
                um_per_pixel = prediction_boxes.new_full((batch_size, 2), 0.125)
            normalized_size, normalized_scale = normalize_image_metadata(
                batch_size,
                prediction_boxes.device,
                prediction_boxes.dtype,
                image_size_px,
                um_per_pixel,
            )
            prediction_dimensions = calibrated_length_width(
                prediction_boxes.unsqueeze(1),
                normalized_size[batch_indices],
                normalized_scale[batch_indices],
            ).squeeze(1)

            target_dimensions: list[Tensor] = []
            cursor = 0
            for batch_index, indices in enumerate(target_indices_by_image):
                if indices.numel() == 0:
                    continue
                count = indices.numel()
                if "length_width_um" in targets[batch_index]:
                    dimensions = targets[batch_index]["length_width_um"][indices]
                else:
                    dimensions = calibrated_length_width(
                        target_boxes[cursor : cursor + count].unsqueeze(1),
                        normalized_size[batch_index : batch_index + 1].expand(
                            count, -1
                        ),
                        normalized_scale[batch_index : batch_index + 1].expand(
                            count, -1
                        ),
                    ).squeeze(1)
                target_dimensions.append(dimensions)
                cursor += count
            loss_morphometry = F.smooth_l1_loss(
                prediction_dimensions,
                torch.cat(target_dimensions).to(prediction_dimensions),
            )

        target_total_counts = torch.stack(
            [
                target["decoration_counts"]
                .to(outputs["auxiliary_decoration_count"])
                .float()
                .sum()
                for target in targets
            ]
        )
        loss_auxiliary_count = F.smooth_l1_loss(
            outputs["auxiliary_decoration_count"], target_total_counts
        )

        total = (
            self.loss_weights.presence * loss_presence
            + self.loss_weights.bbox_l1 * loss_bbox_l1
            + self.loss_weights.bbox_giou * loss_bbox_giou
            + self.loss_weights.morphometry * loss_morphometry
            + self.loss_weights.ge9 * loss_ge9
            + self.loss_weights.target_count * loss_target_count
            + self.loss_weights.auxiliary_count * loss_auxiliary_count
        )
        return {
            "loss": total,
            "loss_presence": loss_presence,
            "loss_bbox_l1": loss_bbox_l1,
            "loss_bbox_giou": loss_bbox_giou,
            "loss_morphometry": loss_morphometry,
            "loss_ge9": loss_ge9,
            "loss_target_count": loss_target_count,
            "loss_auxiliary_count": loss_auxiliary_count,
        }
