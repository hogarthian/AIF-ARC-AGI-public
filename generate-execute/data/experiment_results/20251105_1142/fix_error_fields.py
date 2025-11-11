"""
Fix error fields in results.json files where grids have been fixed but results still have "error" field.

This script:
1. Checks each experiment directory
2. For each entry in results.json that has "error" field
3. Checks if corresponding grids.json shows grids are fixed (has error-fixed or valid grids)
4. If grids are fixed, changes "error" to "error-fixed" in results.json
5. Also updates main results.json
"""

import json
from pathlib import Path
from typing import Dict, Any

def is_valid_grid(grid: Any) -> bool:
    """Check if grid is valid (not None and is a list)."""
    return grid is not None and isinstance(grid, list)

def fix_error_fields_in_experiment(experiment_dir: Path) -> Dict[str, Any]:
    """Fix error fields in a single experiment directory."""
    results_file = experiment_dir / "results.json"
    results_reduced_file = experiment_dir / "results_reduced.json"
    grids_file = experiment_dir / "grids.json"
    grids_reduced_file = experiment_dir / "grids_reduced.json"
    
    stats = {
        "fixed_normal": 0,
        "fixed_reduced": 0,
        "fixed_keys": []
    }
    
    # Process normal results
    if results_file.exists() and grids_file.exists():
        with open(results_file) as f:
            results = json.load(f)
        with open(grids_file) as f:
            grids = json.load(f)
        
        modified = False
        for key in list(results.keys()):
            if "-" in key:  # Skip backup entries
                continue
            
            if not (key.startswith("E") or key.startswith("T")):
                continue
            
            result_entry = results[key]
            grid_entry = grids.get(key, {})
            
            # Only process if result has "error" field
            if "error" not in result_entry:
                continue
            
            # Check if grids are fixed
            has_error_in_grid = "error" in grid_entry
            has_error_fixed_in_grid = "error-fixed" in grid_entry
            has_valid_grids = (
                is_valid_grid(grid_entry.get("ascending")) and 
                is_valid_grid(grid_entry.get("descending"))
            )
            
            # Grid is fixed if it has error-fixed OR (no error and has valid grids)
            grid_is_fixed = has_error_fixed_in_grid or (not has_error_in_grid and has_valid_grids)
            
            if grid_is_fixed:
                # Change error to error-fixed
                error_msg = result_entry["error"]
                result_entry["error-fixed"] = error_msg
                del result_entry["error"]
                modified = True
                stats["fixed_normal"] += 1
                stats["fixed_keys"].append(f"{experiment_dir.name}/{key} (normal)")
                print(f"  Fixed {key} (normal): changed error to error-fixed")
        
        if modified:
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
    
    # Process reduced results
    if results_reduced_file.exists() and grids_reduced_file.exists():
        with open(results_reduced_file) as f:
            results_reduced = json.load(f)
        with open(grids_reduced_file) as f:
            grids_reduced = json.load(f)
        
        modified = False
        for key in list(results_reduced.keys()):
            if "-" in key:  # Skip backup entries
                continue
            
            if not (key.startswith("E") or key.startswith("T")):
                continue
            
            result_entry = results_reduced[key]
            grid_entry = grids_reduced.get(key, {})
            
            # Only process if result has "error" field
            if "error" not in result_entry:
                continue
            
            # Check if grids are fixed
            has_error_in_grid = "error" in grid_entry
            has_error_fixed_in_grid = "error-fixed" in grid_entry
            has_valid_grids = (
                is_valid_grid(grid_entry.get("ascending")) and 
                is_valid_grid(grid_entry.get("descending"))
            )
            
            # Grid is fixed if it has error-fixed OR (no error and has valid grids)
            grid_is_fixed = has_error_fixed_in_grid or (not has_error_in_grid and has_valid_grids)
            
            if grid_is_fixed:
                # Change error to error-fixed
                error_msg = result_entry["error"]
                result_entry["error-fixed"] = error_msg
                del result_entry["error"]
                modified = True
                stats["fixed_reduced"] += 1
                stats["fixed_keys"].append(f"{experiment_dir.name}/{key} (reduced)")
                print(f"  Fixed {key} (reduced): changed error to error-fixed")
        
        if modified:
            with open(results_reduced_file, "w") as f:
                json.dump(results_reduced, f, indent=2)
    
    return stats

def update_main_results(main_results_file: Path):
    """Update main results.json by reloading from individual experiment results."""
    if not main_results_file.exists():
        print(f"  Main results.json not found: {main_results_file}")
        return
    
    experiment_dir = main_results_file.parent
    
    with open(main_results_file) as f:
        main_results = json.load(f)
    
    results_list = main_results.get("results", [])
    updated_count = 0
    
    for result in results_list:
        modality_type = result.get("modality_type")
        order_name = result.get("example_order")
        
        # Find corresponding experiment directory
        exp_dir = experiment_dir / f"{modality_type}_{order_name}"
        
        if not exp_dir.exists():
            continue
        
        # Load updated results
        results_file = exp_dir / "results.json"
        results_reduced_file = exp_dir / "results_reduced.json"
        
        if not results_file.exists() or not results_reduced_file.exists():
            continue
        
        with open(results_file) as f:
            corrected_results = json.load(f)
        
        with open(results_reduced_file) as f:
            corrected_results_reduced = json.load(f)
        
        # Extract held-out results (sort by held_out_idx)
        held_out_results = []
        held_out_results_reduced = []
        
        held_out_dict = {}
        held_out_reduced_dict = {}
        
        for key in corrected_results.keys():
            if key.startswith("E") and "-" not in key:
                held_out_dict[key] = corrected_results[key]
        
        for key in corrected_results_reduced.keys():
            if key.startswith("E") and "-" not in key:
                held_out_reduced_dict[key] = corrected_results_reduced[key]
        
        held_out_results = sorted(
            held_out_dict.values(),
            key=lambda x: x.get("held_out_idx", 0)
        )
        held_out_results_reduced = sorted(
            held_out_reduced_dict.values(),
            key=lambda x: x.get("held_out_idx", 0)
        )
        
        # Extract test results (sort by test_idx)
        test_results = []
        test_results_reduced = []
        
        test_dict = {}
        test_reduced_dict = {}
        
        for key in corrected_results.keys():
            if key.startswith("T") and "-" not in key:
                test_dict[key] = corrected_results[key]
        
        for key in corrected_results_reduced.keys():
            if key.startswith("T") and "-" not in key:
                test_reduced_dict[key] = corrected_results_reduced[key]
        
        test_results = sorted(
            test_dict.values(),
            key=lambda x: x.get("test_idx", 0)
        )
        test_results_reduced = sorted(
            test_reduced_dict.values(),
            key=lambda x: x.get("test_idx", 0)
        )
        
        # Update result entry
        result["held_out_results"] = held_out_results
        result["held_out_results_reduced"] = held_out_results_reduced
        result["test_results"] = test_results
        result["test_results_reduced"] = test_results_reduced
        updated_count += 1
    
    # Save updated main results
    with open(main_results_file, "w") as f:
        json.dump(main_results, f, indent=2)
    
    print(f"\n  Updated main results.json: {updated_count} experiments")

def main():
    experiment_dir = Path(".")
    main_results_file = experiment_dir / "results.json"
    
    print("Fixing error fields in results.json files...")
    print("=" * 80)
    
    all_stats = []
    
    # Process all experiment directories
    for subdir in experiment_dir.iterdir():
        if not subdir.is_dir() or subdir.name == "plots":
            continue
        
        # Check if this is an experiment directory
        if not (subdir / "results.json").exists():
            continue
        
        print(f"\nProcessing: {subdir.name}")
        stats = fix_error_fields_in_experiment(subdir)
        if stats["fixed_normal"] > 0 or stats["fixed_reduced"] > 0:
            all_stats.append(stats)
    
    # Update main results.json
    print("\n" + "=" * 80)
    print("Updating main results.json...")
    update_main_results(main_results_file)
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary:")
    total_fixed = sum(s["fixed_normal"] + s["fixed_reduced"] for s in all_stats)
    print(f"Total entries fixed: {total_fixed}")
    
    if all_stats:
        print("\nFixed entries:")
        for stats in all_stats:
            for key in stats["fixed_keys"]:
                print(f"  - {key}")
    
    print("\nDone! Error fields have been updated to error-fixed where grids are fixed.")
    print("Cross-order analysis should now include these data points.")

if __name__ == "__main__":
    main()

