from typing import List, Dict, Optional, Tuple
import torch
from torchvision.ops import box_convert, box_iou, generalized_box_iou



class AnswerLocalizationEvaluator:
    def __init__(self, iou_thr: float = 0.50, eps: float = 1e-6, pred_fmt: str = "cxcywh", calculate_hist: bool = False, hist_bins: int = 50):
        self.iou_thr = iou_thr
        self.eps = eps
        self.pred_fmt = pred_fmt
        self.calculate_hist = calculate_hist
        self.hist_bins = hist_bins

    def calculate_iou(self, pred_xyxy: torch.Tensor, gt_xyxy: torch.Tensor) -> float:
        iou = box_iou(pred_xyxy.unsqueeze(0), gt_xyxy.unsqueeze(0))[0, 0]
        return float(iou.item())

    def calculate_giou(self, pred_xyxy: torch.Tensor, gt_xyxy: torch.Tensor) -> float:
        giou = generalized_box_iou(pred_xyxy.unsqueeze(0), gt_xyxy.unsqueeze(0))[0, 0]
        return float(giou.item())

    def calculate_recall(self, iou_value: float) -> float:
        return float(iou_value >= self.iou_thr)

    def calculate_center_l2(self, pred_xyxy: torch.Tensor, gt_xyxy: torch.Tensor) -> float:
        p = box_convert(pred_xyxy.unsqueeze(0), in_fmt="xyxy", out_fmt="cxcywh")[0]
        g = box_convert(gt_xyxy.unsqueeze(0), in_fmt="xyxy", out_fmt="cxcywh")[0]
        l2 = torch.sqrt((p[0] - g[0])**2 + (p[1] - g[1])**2)
        return float(l2.item())

    def calculate_region_stats(self, pred_xyxy: torch.Tensor, gt_xyxy: torch.Tensor):
        # Compute intersection
        ix1 = torch.maximum(pred_xyxy[0], gt_xyxy[0])
        iy1 = torch.maximum(pred_xyxy[1], gt_xyxy[1])
        ix2 = torch.minimum(pred_xyxy[2], gt_xyxy[2])
        iy2 = torch.minimum(pred_xyxy[3], gt_xyxy[3])
        iw = torch.clamp(ix2 - ix1, min=0.0)
        ih = torch.clamp(iy2 - iy1, min=0.0)
        inter = iw * ih

        # Compute areas
        area_of_prediction = torch.clamp(pred_xyxy[2] - pred_xyxy[0], min=self.eps) * torch.clamp(pred_xyxy[3] - pred_xyxy[1], min=self.eps)
        area_of_gt = torch.clamp(gt_xyxy[2] - gt_xyxy[0], min=self.eps) * torch.clamp(gt_xyxy[3] - gt_xyxy[1], min=self.eps)

        # Compute metrics
        coverage = inter / area_of_gt
        purity = inter / area_of_prediction
        area_ratio = area_of_prediction / area_of_gt

        return float(coverage.item()), float(purity.item()), float(area_ratio.item())
    
    def compute_hist(self, values: List[float], bins: int, value_range: Tuple[float, float]) -> Tuple[list, list]:
        # Check if it has values 
        if not values:
            edges = torch.linspace(value_range[0], value_range[1], bins + 1)
            counts = torch.zeros(bins, dtype=torch.int64)
            return counts.tolist(), edges.tolist()
        
        # Clamp edges
        t = torch.tensor(values, dtype=torch.float32)
        lo, hi = value_range
        eps = 1e-9
        t = t.clamp(min=lo, max=hi - eps)

        # Compute hist
        counts, edges = torch.histogram(t, bins=bins, range=value_range)
        return counts.to(torch.int64).tolist(), edges.tolist()


    def compute(self, gt_boxes: List[torch.Tensor], pred_boxes: List[torch.Tensor], image_sizes: List[torch.Tensor]) -> Dict[str, float]:

        # Lists to accumulate per-sample metrics
        iou_vals, giou_vals, rec_vals = [], [], []
        center_l2_vals = []
        coverage_vals, purity_vals, area_ratio_vals = [], [], []

        # Process each sample in the batch
        for gt_b, pr_b, sz_b in zip(gt_boxes, pred_boxes, image_sizes):
            # Turn all to tensors and move to CPU
            gt_b = torch.as_tensor(gt_b, dtype=torch.float32, device="cpu")
            pr_b = torch.as_tensor(pr_b, dtype=torch.float32, device="cpu")
            sz_b = torch.as_tensor(sz_b, dtype=torch.float32, device="cpu")

            # Normalize GT xyxy by (W,H)
            gt_max = float(gt_b.max().item()) if gt_b.numel() > 0 else 0.0
            gt_is_normalized = gt_max <= 1.0 + 1e-3
            if gt_is_normalized:
                gt_xyxy = gt_b.clone().float().clamp(0.0, 1.0)
            else:
                W = sz_b[:, 0:1].clamp_min(self.eps)
                H = sz_b[:, 1:2].clamp_min(self.eps)
                gt_xyxy = gt_b.clone().float()
                gt_xyxy[:, [0, 2]] = (gt_xyxy[:, [0, 2]] / W).clamp_min(self.eps)
                gt_xyxy[:, [1, 3]] = (gt_xyxy[:, [1, 3]] / H).clamp_min(self.eps)


            # Convert preds to xyxy
            pr_b = pr_b.float().clamp(0.0, 1.0)
            if self.pred_fmt == "cxcywh":
                pred_xyxy = box_convert(pr_b, in_fmt="cxcywh", out_fmt="xyxy")
            elif self.pred_fmt == "xyxy":
                pred_xyxy = pr_b
            else:
                raise ValueError(f"Unsupported pred_fmt={self.pred_fmt}")

            # Per-sample metrics
            for i in range(gt_xyxy.size(0)):
                # Get the example boxes
                g = gt_xyxy[i]
                p = pred_xyxy[i]

                # Compute metrics
                iou = self.calculate_iou(p, g)
                giou = self.calculate_giou(p, g)
                rec = self.calculate_recall(iou)
                c_l2 = self.calculate_center_l2(p, g)
                cov, pur, ar = self.calculate_region_stats(p, g)

                # Accumulate
                iou_vals.append(iou)
                giou_vals.append(giou)
                rec_vals.append(rec)
                center_l2_vals.append(c_l2)
                coverage_vals.append(cov)
                purity_vals.append(pur)
                area_ratio_vals.append(ar)

        # Means over all samples
        def mean(xs): 
            return float(torch.tensor(xs, dtype=torch.float32).mean().item()) if len(xs) else 0.0

        # Median IoU 
        iou_median = float(torch.tensor(iou_vals, dtype=torch.float32).median().item()) if len(iou_vals) else 0.0

        # Log the IoU, Pur, Cov as histogram buckets, and add them to the output dict
        hists: Optional[Dict[str, Dict[str, list]]] = None
        if self.calculate_hist:
            # Creat empty history dict 
            hists = {
                "iou": { "counts": None, "edges": None },
                "coverage": {"counts": None, "edges": None},
                "purity": {"counts": None, "edges": None},
            }

            # IoU hist
            c, e = self.compute_hist(iou_vals, self.hist_bins, (0.0, 1.0))
            hists["iou"]["counts"], hists["iou"]["edges"] = c, e

            # Coverage hist
            c, e = self.compute_hist(coverage_vals, self.hist_bins, (0.0, 1.0))
            hists["coverage"]["counts"], hists["coverage"]["edges"] = c, e

            # Purity histogram
            c, e = self.compute_hist(purity_vals, self.hist_bins, (0.0, 1.0))
            hists["purity"]["counts"], hists["purity"]["edges"] = c, e
        
        # Return metrics as a dictionary
        return {
            "iou_mean": mean(iou_vals),
            "iou_median": iou_median,
            "giou_mean": mean(giou_vals),
            "recall@th": mean(rec_vals),
            "center_l2_mean": mean(center_l2_vals),
            "coverage_mean": mean(coverage_vals),
            "purity_mean": mean(purity_vals),
            "area_ratio_mean": mean(area_ratio_vals),
            "hists": hists,
        }
