import torch


class QuestionPriorEvaluator:
    def __init__(self, top_k_percent=0.3, sparsity_threshold=0.01, eps=1e-6):
        self.top_k_percent = top_k_percent
        self.sparsity_threshold = sparsity_threshold
        self.eps = eps

    # Soft IoU
    def soft_iou(self, prior_heatmap, gt_bbox_xyxy):
        """
        Compute the soft Intersection over Union (IoU) between the prior heatmap and the ground truth bounding box.
        """
        x1, y1, x2, y2 = gt_bbox_xyxy
        H, W = prior_heatmap.shape

        # Create GT mask
        gt_mask = torch.zeros((H, W), dtype=torch.float32)
        gt_mask[y1:y2, x1:x2] = 1.0

        # Compute soft IoU
        intersection = (prior_heatmap * gt_mask).sum()
        union = prior_heatmap.sum() + gt_mask.sum() - intersection + self.eps 

        return intersection / union

    # Precision@K: Of the top K% highest prior regions, how much overlaps with GT?
    def precision_at_k(self, prior_heatmap, gt_bbox_xyxy):
        """
        Compute Precision@K between the prior heatmap and the ground truth bounding box.
        """
        x1, y1, x2, y2 = gt_bbox_xyxy
        H, W = prior_heatmap.shape

        # Create GT mask
        gt_mask = torch.zeros((H, W), dtype=torch.float32)
        gt_mask[y1:y2, x1:x2] = 1.0

        # Determine threshold for top K%
        k = int(self.top_k_percent * H * W)
        threshold = torch.topk(prior_heatmap.view(-1), k).values[-1]

        # Create top K% mask
        top_k_mask = (prior_heatmap >= threshold).float()

        # Compute Precision@K
        true_positives = (top_k_mask * gt_mask).sum()
        predicted_positives = top_k_mask.sum() + self.eps  

        return true_positives / predicted_positives

    # Recall@K: Of the GT region, how much is in the top K% of prior?
    def recall_at_k(self, prior_heatmap, gt_bbox_xyxy):
        """
        Compute Recall@K between the prior heatmap and the ground truth bounding box.
        """
        x1, y1, x2, y2 = gt_bbox_xyxy
        H, W = prior_heatmap.shape

        # Create GT mask
        gt_mask = torch.zeros((H, W), dtype=torch.float32)
        gt_mask[y1:y2, x1:x2] = 1.0

        # Determine threshold for top K%
        k = int(self.top_k_percent * H * W)
        threshold = torch.topk(prior_heatmap.view(-1), k).values[-1]

        # Create top K% mask
        top_k_mask = (prior_heatmap >= threshold).float()

        # Compute Recall@K
        true_positives = (top_k_mask * gt_mask).sum()
        actual_positives = gt_mask.sum() + self.eps  

        return true_positives / actual_positives

    # Sparsity of Heatmap
    def sparsity(self, prior_heatmap):
        """
        Fraction of pixels with negligible activation.
        """
        H, W = prior_heatmap.shape
        active_pixels = (prior_heatmap > self.sparsity_threshold).sum()
        sparsity = 1.0 - (active_pixels / (H * W))
        return sparsity 

    # Jensen-Shannon Divergence
    def jensen_shannon_divergence(self, prior_heatmap, gt_bbox_xyxy):
        x1, y1, x2, y2 = gt_bbox_xyxy
        H, W = prior_heatmap.shape

        # Create GT distribution
        gt_distribution = torch.zeros((H, W), dtype=torch.float32)
        gt_distribution[y1:y2, x1:x2] = 1.0
        gt_distribution = gt_distribution / (gt_distribution.sum() + self.eps)

        # Normalize prior heatmap
        prior_distribution = prior_heatmap / (prior_heatmap.sum() + self.eps)

        # Add epsilon for numerical stability
        prior_distribution = prior_distribution + self.eps
        gt_distribution = gt_distribution + self.eps
        
        # Renormalize after adding epsilon
        prior_distribution = prior_distribution / prior_distribution.sum()
        gt_distribution = gt_distribution / gt_distribution.sum()

        # Compute midpoint
        m = 0.5 * (prior_distribution + gt_distribution)

        # Compute KL divergences
        kl_prior_m = (prior_distribution * torch.log(prior_distribution / m)).sum()
        kl_gt_m = (gt_distribution * torch.log(gt_distribution / m)).sum()

        # Jensen-Shannon Divergence
        jsd = 0.5 * (kl_prior_m + kl_gt_m)

        return jsd
    

    def evaluate(self, prior_heatmap, gt_bbox_xyxy):
        """
        Evaluate all metrics between the prior heatmap and the ground truth bounding box.
        Returns a dictionary of all computed metrics.
        """
        return {
            "soft_iou": self.soft_iou(prior_heatmap, gt_bbox_xyxy),
            "precision_at_k": self.precision_at_k(prior_heatmap, gt_bbox_xyxy),
            "recall_at_k": self.recall_at_k(prior_heatmap, gt_bbox_xyxy),
            "sparsity": self.sparsity(prior_heatmap),
            "jensen_shannon_divergence": self.jensen_shannon_divergence(prior_heatmap, gt_bbox_xyxy),
        }