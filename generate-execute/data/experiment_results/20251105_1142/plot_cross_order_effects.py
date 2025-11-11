"""
Plot cross-order effects: When instructions are created with one order (asc/desc),
how do they perform when tested with the same order vs the opposite order?

This script analyzes the existing experiment data to show:
1. Instructions created with ascending order: asc test scores vs desc test scores
2. Instructions created with descending order: asc test scores vs desc test scores
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np

MODALITY_TYPES = [
    "row_only",
    "col_only",
    "image_only",
    "row_col",
    "row_image",
    "col_image",
    "row_col_image",
]


def load_results(results_file: Path) -> Dict[str, Any]:
    """Load results.json file."""
    with open(results_file, "r") as f:
        return json.load(f)


def extract_cross_order_scores(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[Tuple[float, float]]]]:
    """
    Extract cross-order scores from results.
    
    Returns:
        Dict with structure:
        {
            modality: {
                "held_out": [(asc_score, desc_score), ...],  # For each held-out example
                "test": [(asc_score, desc_score), ...],      # For each test case
                "held_out_reduced": [...],
                "test_reduced": [...]
            }
        }
    """
    cross_order_data = {}
    
    for result in results:
        if "error" in result:
            continue
        
        modality = result["modality_type"]
        hypothesis_order = result["example_order"]  # Order used to GENERATE hypothesis
        
        if modality not in cross_order_data:
            cross_order_data[modality] = {
                "held_out": [],
                "test": [],
                "held_out_reduced": [],
                "test_reduced": [],
                "hypothesis_order": {}  # Track which order was used for hypothesis generation
            }
        
        # Store hypothesis order info
        cross_order_data[modality]["hypothesis_order"][hypothesis_order] = True
        
        # Extract held-out results (normal)
        for ho_result in result.get("held_out_results", []):
            if "error" not in ho_result:
                asc_score = ho_result.get("ascending", {}).get("similarity", 0.0)
                desc_score = ho_result.get("descending", {}).get("similarity", 0.0)
                cross_order_data[modality]["held_out"].append({
                    "asc_score": asc_score,
                    "desc_score": desc_score,
                    "hypothesis_order": hypothesis_order,
                    "held_out_idx": ho_result.get("held_out_idx")
                })
        
        # Extract test results (normal)
        for test_result in result.get("test_results", []):
            if "error" not in test_result:
                asc_score = test_result.get("ascending", {}).get("similarity", 0.0)
                desc_score = test_result.get("descending", {}).get("similarity", 0.0)
                cross_order_data[modality]["test"].append({
                    "asc_score": asc_score,
                    "desc_score": desc_score,
                    "hypothesis_order": hypothesis_order,
                    "test_idx": test_result.get("test_idx")
                })
        
        # Extract held-out results (reduced)
        for ho_result in result.get("held_out_results_reduced", []):
            if "error" not in ho_result:
                asc_score = ho_result.get("ascending", {}).get("similarity", 0.0)
                desc_score = ho_result.get("descending", {}).get("similarity", 0.0)
                cross_order_data[modality]["held_out_reduced"].append({
                    "asc_score": asc_score,
                    "desc_score": desc_score,
                    "hypothesis_order": hypothesis_order,
                    "held_out_idx": ho_result.get("held_out_idx")
                })
        
        # Extract test results (reduced)
        for test_result in result.get("test_results_reduced", []):
            if "error" not in test_result:
                asc_score = test_result.get("ascending", {}).get("similarity", 0.0)
                desc_score = test_result.get("descending", {}).get("similarity", 0.0)
                cross_order_data[modality]["test_reduced"].append({
                    "asc_score": asc_score,
                    "desc_score": desc_score,
                    "hypothesis_order": hypothesis_order,
                    "test_idx": test_result.get("test_idx")
                })
    
    return cross_order_data


def create_cross_order_plot(
    cross_order_data: Dict[str, Dict[str, List[Dict[str, Any]]]],
    plots_dir: Path,
    variant: str = "normal"
):
    """
    Create plot showing cross-order effects.
    
    Args:
        cross_order_data: Extracted cross-order data
        plots_dir: Directory to save plots
        variant: "normal" or "reduced"
    """
    # Filter to available modalities
    available_modalities = [m for m in MODALITY_TYPES if m in cross_order_data]
    
    if not available_modalities:
        print(f"No data available for {variant} variant")
        return
    
    # Create subplots: one for held-out, one for tests
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Held-out examples
    for modality in available_modalities:
        data = cross_order_data[modality]
        key = "held_out" if variant == "normal" else "held_out_reduced"
        results = data.get(key, [])
        
        if not results:
            continue
        
        # Separate by hypothesis order
        asc_hyp_results = [r for r in results if r["hypothesis_order"] == "ascending"]
        desc_hyp_results = [r for r in results if r["hypothesis_order"] == "descending"]
        
        # Plot ascending hypothesis: asc test vs desc test
        if asc_hyp_results:
            asc_test_scores = [r["asc_score"] for r in asc_hyp_results]
            desc_test_scores = [r["desc_score"] for r in asc_hyp_results]
            indices = [r["held_out_idx"] for r in asc_hyp_results]
            
            # Sort by index for clean lines
            sorted_data = sorted(zip(indices, asc_test_scores, desc_test_scores))
            indices, asc_scores, desc_scores = zip(*sorted_data) if sorted_data else ([], [], [])
            
            if asc_scores:
                ax1.plot(
                    indices, asc_scores,
                    color='blue', linestyle='-', marker='o', markersize=5,
                    alpha=0.7, linewidth=2,
                    label=f"{modality} (asc hyp → asc test)"
                )
                ax1.plot(
                    indices, desc_scores,
                    color='lightblue', linestyle='--', marker='s', markersize=5,
                    alpha=0.7, linewidth=2,
                    label=f"{modality} (asc hyp → desc test)"
                )
        
        # Plot descending hypothesis: asc test vs desc test
        if desc_hyp_results:
            asc_test_scores = [r["asc_score"] for r in desc_hyp_results]
            desc_test_scores = [r["desc_score"] for r in desc_hyp_results]
            indices = [r["held_out_idx"] for r in desc_hyp_results]
            
            # Sort by index
            sorted_data = sorted(zip(indices, asc_test_scores, desc_test_scores))
            indices, asc_scores, desc_scores = zip(*sorted_data) if sorted_data else ([], [], [])
            
            if asc_scores:
                ax1.plot(
                    indices, asc_scores,
                    color='red', linestyle='-', marker='o', markersize=5,
                    alpha=0.7, linewidth=2,
                    label=f"{modality} (desc hyp → asc test)"
                )
                ax1.plot(
                    indices, desc_scores,
                    color='coral', linestyle='--', marker='s', markersize=5,
                    alpha=0.7, linewidth=2,
                    label=f"{modality} (desc hyp → desc test)"
                )
    
    ax1.set_xlabel("Held-Out Example Index", fontsize=12)
    ax1.set_ylabel("Similarity Score", fontsize=12)
    ax1.set_title(f"Held-Out Validation: Cross-Order Effects ({variant})", fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1.05])
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=7, ncol=2)
    
    # Plot 2: Test cases
    for modality in available_modalities:
        data = cross_order_data[modality]
        key = "test" if variant == "normal" else "test_reduced"
        results = data.get(key, [])
        
        if not results:
            continue
        
        # Separate by hypothesis order
        asc_hyp_results = [r for r in results if r["hypothesis_order"] == "ascending"]
        desc_hyp_results = [r for r in results if r["hypothesis_order"] == "descending"]
        
        # Plot ascending hypothesis: asc test vs desc test
        if asc_hyp_results:
            asc_test_scores = [r["asc_score"] for r in asc_hyp_results]
            desc_test_scores = [r["desc_score"] for r in asc_hyp_results]
            indices = [r["test_idx"] for r in asc_hyp_results]
            
            # Sort by index
            sorted_data = sorted(zip(indices, asc_test_scores, desc_test_scores))
            indices, asc_scores, desc_scores = zip(*sorted_data) if sorted_data else ([], [], [])
            
            if asc_scores:
                ax2.plot(
                    indices, asc_scores,
                    color='blue', linestyle='-', marker='o', markersize=5,
                    alpha=0.7, linewidth=2,
                    label=f"{modality} (asc hyp → asc test)"
                )
                ax2.plot(
                    indices, desc_scores,
                    color='lightblue', linestyle='--', marker='s', markersize=5,
                    alpha=0.7, linewidth=2,
                    label=f"{modality} (asc hyp → desc test)"
                )
        
        # Plot descending hypothesis: asc test vs desc test
        if desc_hyp_results:
            asc_test_scores = [r["asc_score"] for r in desc_hyp_results]
            desc_test_scores = [r["desc_score"] for r in desc_hyp_results]
            indices = [r["test_idx"] for r in desc_hyp_results]
            
            # Sort by index
            sorted_data = sorted(zip(indices, asc_test_scores, desc_test_scores))
            indices, asc_scores, desc_scores = zip(*sorted_data) if sorted_data else ([], [], [])
            
            if asc_scores:
                ax2.plot(
                    indices, asc_scores,
                    color='red', linestyle='-', marker='o', markersize=5,
                    alpha=0.7, linewidth=2,
                    label=f"{modality} (desc hyp → asc test)"
                )
                ax2.plot(
                    indices, desc_scores,
                    color='coral', linestyle='--', marker='s', markersize=5,
                    alpha=0.7, linewidth=2,
                    label=f"{modality} (desc hyp → desc test)"
                )
    
    ax2.set_xlabel("Test Case Index", fontsize=12)
    ax2.set_ylabel("Similarity Score", fontsize=12)
    ax2.set_title(f"Test Cases: Cross-Order Effects ({variant})", fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 1.05])
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=7, ncol=2)
    
    plt.tight_layout()
    filename = f"cross_order_effects_{variant}.png"
    plt.savefig(plots_dir / filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved cross-order effects plot: {plots_dir / filename}")


def create_scatter_plot(
    cross_order_data: Dict[str, Dict[str, List[Dict[str, Any]]]],
    plots_dir: Path,
    variant: str = "normal"
):
    """
    Create scatter plot: x=ascending test score, y=descending test score.
    Points closer to diagonal line indicate less cross-order effect.
    """
    available_modalities = [m for m in MODALITY_TYPES if m in cross_order_data]
    
    if not available_modalities:
        print(f"No data available for {variant} variant scatter plot")
        return
    
    fig, axes = plt.subplots(2, len(available_modalities), figsize=(4*len(available_modalities), 8))
    if len(available_modalities) == 1:
        axes = axes.reshape(2, 1)
    
    # Row 1: Held-out examples
    # Row 2: Test cases
    
    for col_idx, modality in enumerate(available_modalities):
        data = cross_order_data[modality]
        
        # Plot held-out (row 0)
        ax1 = axes[0, col_idx]
        key = "held_out" if variant == "normal" else "held_out_reduced"
        results = data.get(key, [])
        
        if results:
            asc_hyp_results = [r for r in results if r["hypothesis_order"] == "ascending"]
            desc_hyp_results = [r for r in results if r["hypothesis_order"] == "descending"]
            
            if asc_hyp_results:
                asc_scores = [r["asc_score"] for r in asc_hyp_results]
                desc_scores = [r["desc_score"] for r in asc_hyp_results]
                ax1.scatter(asc_scores, desc_scores, alpha=0.6, s=50, color='blue', 
                           label=f"Asc hyp (n={len(asc_scores)})", marker='o')
            
            if desc_hyp_results:
                asc_scores = [r["asc_score"] for r in desc_hyp_results]
                desc_scores = [r["desc_score"] for r in desc_hyp_results]
                ax1.scatter(asc_scores, desc_scores, alpha=0.6, s=50, color='red',
                           label=f"Desc hyp (n={len(desc_scores)})", marker='s')
            
            # Add diagonal line
            ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
            ax1.set_xlabel("Ascending Test Score", fontsize=10)
            ax1.set_ylabel("Descending Test Score", fontsize=10)
            ax1.set_title(f"{modality}\nHeld-Out ({variant})", fontsize=11, fontweight='bold')
            ax1.set_xlim([0, 1.05])
            ax1.set_ylim([0, 1.05])
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='best', fontsize=8)
        
        # Plot test cases (row 1)
        ax2 = axes[1, col_idx]
        key = "test" if variant == "normal" else "test_reduced"
        results = data.get(key, [])
        
        if results:
            asc_hyp_results = [r for r in results if r["hypothesis_order"] == "ascending"]
            desc_hyp_results = [r for r in results if r["hypothesis_order"] == "descending"]
            
            if asc_hyp_results:
                asc_scores = [r["asc_score"] for r in asc_hyp_results]
                desc_scores = [r["desc_score"] for r in asc_hyp_results]
                ax2.scatter(asc_scores, desc_scores, alpha=0.6, s=50, color='blue',
                           label=f"Asc hyp (n={len(asc_scores)})", marker='o')
            
            if desc_hyp_results:
                asc_scores = [r["asc_score"] for r in desc_hyp_results]
                desc_scores = [r["desc_score"] for r in desc_hyp_results]
                ax2.scatter(asc_scores, desc_scores, alpha=0.6, s=50, color='red',
                           label=f"Desc hyp (n={len(desc_scores)})", marker='s')
            
            # Add diagonal line
            ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
            ax2.set_xlabel("Ascending Test Score", fontsize=10)
            ax2.set_ylabel("Descending Test Score", fontsize=10)
            ax2.set_title(f"{modality}\nTest Cases ({variant})", fontsize=11, fontweight='bold')
            ax2.set_xlim([0, 1.05])
            ax2.set_ylim([0, 1.05])
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='best', fontsize=8)
    
    plt.tight_layout()
    filename = f"cross_order_scatter_{variant}.png"
    plt.savefig(plots_dir / filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved cross-order scatter plot: {plots_dir / filename}")


def create_summary_statistics(
    cross_order_data: Dict[str, Dict[str, List[Dict[str, Any]]]],
    plots_dir: Path
):
    """Create summary statistics table showing cross-order differences."""
    
    stats = []
    
    for modality in MODALITY_TYPES:
        if modality not in cross_order_data:
            continue
        
        data = cross_order_data[modality]
        
        for variant in ["held_out", "test", "held_out_reduced", "test_reduced"]:
            results = data.get(variant, [])
            if not results:
                continue
            
            # Separate by hypothesis order
            asc_hyp_results = [r for r in results if r["hypothesis_order"] == "ascending"]
            desc_hyp_results = [r for r in results if r["hypothesis_order"] == "descending"]
            
            # Calculate statistics for ascending hypothesis
            if asc_hyp_results:
                asc_asc_scores = [r["asc_score"] for r in asc_hyp_results]
                asc_desc_scores = [r["desc_score"] for r in asc_hyp_results]
                diff_asc_hyp = [a - d for a, d in zip(asc_asc_scores, asc_desc_scores)]
                
                stats.append({
                    "modality": modality,
                    "variant": variant,
                    "hypothesis_order": "ascending",
                    "n": len(asc_hyp_results),
                    "mean_asc_test": np.mean(asc_asc_scores),
                    "mean_desc_test": np.mean(asc_desc_scores),
                    "mean_diff": np.mean(diff_asc_hyp),
                    "std_diff": np.std(diff_asc_hyp),
                    "max_diff": max(diff_asc_hyp) if diff_asc_hyp else 0,
                    "min_diff": min(diff_asc_hyp) if diff_asc_hyp else 0
                })
            
            # Calculate statistics for descending hypothesis
            if desc_hyp_results:
                desc_asc_scores = [r["asc_score"] for r in desc_hyp_results]
                desc_desc_scores = [r["desc_score"] for r in desc_hyp_results]
                diff_desc_hyp = [a - d for a, d in zip(desc_asc_scores, desc_desc_scores)]
                
                stats.append({
                    "modality": modality,
                    "variant": variant,
                    "hypothesis_order": "descending",
                    "n": len(desc_hyp_results),
                    "mean_asc_test": np.mean(desc_asc_scores),
                    "mean_desc_test": np.mean(desc_desc_scores),
                    "mean_diff": np.mean(diff_desc_hyp),
                    "std_diff": np.std(diff_desc_hyp),
                    "max_diff": max(diff_desc_hyp) if diff_desc_hyp else 0,
                    "min_diff": min(diff_desc_hyp) if diff_desc_hyp else 0
                })
    
    # Save statistics
    stats_file = plots_dir / "cross_order_statistics.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"Saved statistics: {stats_file}")
    
    # Print summary
    print("\n=== Cross-Order Effect Summary ===")
    print("Positive diff = ascending test score > descending test score")
    print("Negative diff = ascending test score < descending test score\n")
    
    for stat in stats:
        print(f"{stat['modality']} ({stat['variant']}), {stat['hypothesis_order']} hyp:")
        print(f"  n={stat['n']}, mean_diff={stat['mean_diff']:.4f}, std={stat['std_diff']:.4f}")
        print(f"  asc_test={stat['mean_asc_test']:.4f}, desc_test={stat['mean_desc_test']:.4f}")
        print()


def main():
    results_file = Path(__file__).parent / "results.json"
    plots_dir = Path(__file__).parent / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    print(f"Loading results from: {results_file}")
    summary = load_results(results_file)
    
    results = summary.get("results", [])
    print(f"Found {len(results)} experiment results")
    
    # Extract cross-order data
    cross_order_data = extract_cross_order_scores(results)
    
    print(f"\nAvailable modalities: {list(cross_order_data.keys())}")
    
    # Create plots for normal variant
    create_cross_order_plot(cross_order_data, plots_dir, variant="normal")
    create_scatter_plot(cross_order_data, plots_dir, variant="normal")
    
    # Create plots for reduced variant
    create_cross_order_plot(cross_order_data, plots_dir, variant="reduced")
    create_scatter_plot(cross_order_data, plots_dir, variant="reduced")
    
    # Create summary statistics
    create_summary_statistics(cross_order_data, plots_dir)
    
    print(f"\nAll plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()

