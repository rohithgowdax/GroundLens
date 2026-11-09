from typing import List
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_rate_vs_anls_scatter(
    rates: List[float],
    anls_scores: List[float],
    output_path: str,
    title: str = "Rate vs ANLS",
    dpi: int = 300,
) -> None:
    """
    Create scatter plot of rate_norm (x) vs anls (y).

    Args:
        rates: List of rate values (one per sample)
        anls_scores: List of ANLS values (one per sample)
        output_path: Path to save the figure
        title: Plot title
        dpi: Resolution for saved figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(rates, anls_scores, alpha=0.3, s=10, c='blue', edgecolors='none')

    ax.set_xlabel("Normalized Rate (KL per patch)", fontsize=12)
    ax.set_ylabel("ANLS", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_rate_vs_anls_binned(
    rates: List[float],
    anls_scores: List[float],
    output_path: str,
    n_bins: int = 50,
    title: str = "Rate vs ANLS",
    dpi: int = 300,
) -> None:
    """
    Create binned curve with equal-count bins and 95% CI error bars.

    Args:
        rates: List of rate values
        anls_scores: List of ANLS values
        output_path: Path to save the figure
        n_bins: Number of equal-count bins
        title: Plot title
        dpi: Resolution for saved figure
    """
    rates_arr = np.array(rates)
    anls_arr = np.array(anls_scores)

    # Sort by rate
    sort_idx = np.argsort(rates_arr)
    rates_sorted = rates_arr[sort_idx]
    anls_sorted = anls_arr[sort_idx]

    # Create equal-count bins
    n_samples = len(rates_arr)
    bin_size = n_samples // n_bins

    bin_centers = []
    bin_means = []
    bin_sems = []

    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size if i < n_bins - 1 else n_samples

        bin_rates = rates_sorted[start_idx:end_idx]
        bin_anls = anls_sorted[start_idx:end_idx]

        if len(bin_anls) > 0:
            bin_centers.append(np.mean(bin_rates))
            bin_means.append(np.mean(bin_anls))
            bin_sems.append(np.std(bin_anls) / np.sqrt(len(bin_anls)))

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.errorbar(
        bin_centers, bin_means,
        yerr=np.array(bin_sems) * 1.96,
        fmt='o-',
        capsize=3,
        capthick=1,
        markersize=6,
        color='blue',
        ecolor='lightblue',
        label='Mean ANLS (95% CI)'
    )

    ax.set_xlabel("Normalized Rate (KL per patch)", fontsize=12)
    ax.set_ylabel("Mean ANLS", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    ax.text(
        0.02, 0.02,
        f"n={len(rates)}, {n_bins} bins",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='bottom'
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_rate_anls(
    rates: List[float],
    anls_scores: List[float],
    output_dir: str,
    n_bins: int = 20,
    dpi: int = 600,
) -> None:
    """
    Generate both scatter and binned Rate vs ANLS plots.

    Args:
        rates: List of rate values (one per sample)
        anls_scores: List of ANLS values (one per sample)
        output_dir: Directory to save figures
        n_bins: Number of bins for binned plot
        dpi: Resolution for saved figures
    """
    scatter_path = os.path.join(output_dir, "rate_vs_anls_scatter.png")
    plot_rate_vs_anls_scatter(rates, anls_scores, scatter_path, dpi=dpi)

    binned_path = os.path.join(output_dir, "rate_vs_anls_binned.png")
    plot_rate_vs_anls_binned(rates, anls_scores, binned_path, n_bins=n_bins, dpi=dpi)


def plot_joint_rate_heatmap(
    rq_values: List[float],
    ra_values: List[float],
    anls_scores: List[float],
    output_dir: str,
    n_bins: int = 10,
    dpi: int = 300,
) -> dict:
    """
    Create a 2D heatmap showing mean ANLS for joint (Rq, Ra) bins.

    Uses quantile-based (equal-count) bins so each row/column has sufficient samples.
    Also computes quadrant summaries using median thresholds.

    Args:
        rq_values: List of question-mask rate values (Rq) - one per sample
        ra_values: List of answer-region rate values (Ra) - one per sample
        anls_scores: List of ANLS values - one per sample
        output_dir: Directory to save figures
        n_bins: Number of bins per axis (creates n_bins x n_bins grid)
        dpi: Resolution for saved figures

    Returns:
        Dict with quadrant statistics and bin edges
    """
    rq_arr = np.array(rq_values)
    ra_arr = np.array(ra_values)
    anls_arr = np.array(anls_scores)
    n_samples = len(rq_arr)

    # Compute quantile-based bin edges (equal-count bins)
    rq_quantiles = np.linspace(0, 100, n_bins + 1)
    ra_quantiles = np.linspace(0, 100, n_bins + 1)

    rq_edges = np.percentile(rq_arr, rq_quantiles)
    ra_edges = np.percentile(ra_arr, ra_quantiles)

    # Assign samples to bins (clip to ensure exactly n_bins bins: 0 to n_bins-1)
    rq_bins = np.clip(np.digitize(rq_arr, rq_edges[1:-1]), 0, n_bins - 1)
    ra_bins = np.clip(np.digitize(ra_arr, ra_edges[1:-1]), 0, n_bins - 1)

    actual_rq_bins = n_bins
    actual_ra_bins = n_bins

    # Build 2D grid of mean ANLS and counts
    mean_anls_grid = np.full((actual_ra_bins, actual_rq_bins), np.nan)
    count_grid = np.zeros((actual_ra_bins, actual_rq_bins), dtype=int)

    for i in range(actual_ra_bins):
        for j in range(actual_rq_bins):
            mask = (rq_bins == j) & (ra_bins == i)
            count = mask.sum()
            count_grid[i, j] = count
            if count > 0:
                mean_anls_grid[i, j] = anls_arr[mask].mean()

    # === Quadrant Analysis using medians ===
    rq_median = np.median(rq_arr)
    ra_median = np.median(ra_arr)

    quadrants = {
        'low_rq_low_ra': (rq_arr < rq_median) & (ra_arr < ra_median),
        'low_rq_high_ra': (rq_arr < rq_median) & (ra_arr >= ra_median),
        'high_rq_low_ra': (rq_arr >= rq_median) & (ra_arr < ra_median),
        'high_rq_high_ra': (rq_arr >= rq_median) & (ra_arr >= ra_median),
    }

    quadrant_stats = {}
    for name, mask in quadrants.items():
        count = mask.sum()
        mean_anls = anls_arr[mask].mean() if count > 0 else 0.0
        std_anls = anls_arr[mask].std() if count > 0 else 0.0
        quadrant_stats[name] = {
            'count': int(count),
            'mean_anls': float(mean_anls),
            'std_anls': float(std_anls),
        }

    # Create quadrant summary figure
    fig2, ax2 = plt.subplots(figsize=(8, 6))

    quadrant_grid = np.array([
        [quadrant_stats['low_rq_low_ra']['mean_anls'], quadrant_stats['high_rq_low_ra']['mean_anls']],
        [quadrant_stats['low_rq_high_ra']['mean_anls'], quadrant_stats['high_rq_high_ra']['mean_anls']],
    ])
    quadrant_counts = np.array([
        [quadrant_stats['low_rq_low_ra']['count'], quadrant_stats['high_rq_low_ra']['count']],
        [quadrant_stats['low_rq_high_ra']['count'], quadrant_stats['high_rq_high_ra']['count']],
    ])

    im2 = ax2.imshow(quadrant_grid, cmap='RdYlGn', vmin=0.0, vmax=1.0, origin='lower')
    plt.colorbar(im2, ax=ax2, label='Mean ANLS', shrink=0.6)

    # Annotate quadrants
    labels = [
        ['Low R(q)\nLow R(a)', 'High R(q)\nLow R(a)'],
        ['Low R(q)\nHigh R(a)', 'High R(q)\nHigh R(a)'],
    ]
    for i in range(2):
        for j in range(2):
            val = quadrant_grid[i, j]
            count = quadrant_counts[i, j]
            text_color = 'white' if val < 0.5 else 'black'
            ax2.text(
                j, i,
                f'{labels[i][j]}\n\nANLS: {val:.3f}\nn={count}',
                ha='center', va='center',
                fontsize=11, color=text_color,
                fontweight='bold'
            )

    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Low R(q)', 'High R(q)'], fontsize=12)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Low R(a)', 'High R(a)'], fontsize=12)
    plt.tight_layout()

    quadrant_path = os.path.join(output_dir, "joint_rate_quadrant_analysis.png")
    plt.savefig(quadrant_path, dpi=dpi, bbox_inches='tight')

    return {
        'rq_median': float(rq_median),
        'ra_median': float(ra_median),
        'rq_edges': rq_edges.tolist(),
        'ra_edges': ra_edges.tolist(),
        'quadrant_stats': quadrant_stats,
        'mean_anls_grid': mean_anls_grid.tolist(),
        'count_grid': count_grid.tolist(),
    }
