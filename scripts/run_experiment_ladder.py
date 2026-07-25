#!/usr/bin/env python3
"""
Experiment ladder runner.

Runs PCA → ConvAE → CRA5 experiments sequentially and compares results.
Implements early stopping when larger training sets don't improve validation RMSE.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentResult:
    """Result of a single experiment."""
    name: str
    type: str
    config_path: str
    success: bool
    metrics: dict[str, Any] | None = None
    error: str | None = None
    runtime_seconds: float = 0.0
    output_dir: Path | None = None


class ExperimentLadder:
    """Manages the experiment ladder execution."""
    
    def __init__(self, config_path: Path):
        """Initialize with ladder configuration."""
        self.config_path = config_path
        self.config = self._load_config()
        self.results: list[ExperimentResult] = []
        
        # Setup output directory
        self.output_dir = Path(self.config["metadata"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Experiment ladder: {self.config['metadata']['name']}")
        print(f"Output directory: {self.output_dir}")
    
    def _load_config(self) -> dict[str, Any]:
        """Load ladder configuration."""
        with self.config_path.open() as f:
            return yaml.safe_load(f)
    
    def _run_pca_experiment(self, exp_config: dict[str, Any]) -> ExperimentResult:
        """Run a PCA baseline experiment."""
        print(f"Running PCA experiment: {exp_config['name']}")
        
        config_path = Path(exp_config["config_path"])
        if not config_path.exists():
            return ExperimentResult(
                name=exp_config["name"],
                type=exp_config["type"],
                config_path=str(config_path),
                success=False,
                error=f"Config file not found: {config_path}",
            )
        
        try:
            start_time = time.perf_counter()
            
            # Run PCA baseline script
            cmd = [
                "python", "scripts/fit_pca_baseline.py",
                "--config", str(config_path),
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config["runtime"]["timeout_minutes"] * 60,
            )
            
            runtime = time.perf_counter() - start_time
            
            if result.returncode != 0:
                return ExperimentResult(
                    name=exp_config["name"],
                    type=exp_config["type"], 
                    config_path=str(config_path),
                    success=False,
                    error=f"PCA script failed: {result.stderr}",
                    runtime_seconds=runtime,
                )
            
            # Parse metrics from output (simplified)
            # In real implementation, this would parse the actual output format
            metrics = {
                "experiment_type": "pca",
                "config_path": str(config_path),
                "success": True,
                "stdout": result.stdout,
            }
            
            return ExperimentResult(
                name=exp_config["name"],
                type=exp_config["type"],
                config_path=str(config_path),
                success=True,
                metrics=metrics,
                runtime_seconds=runtime,
            )
            
        except subprocess.TimeoutExpired:
            return ExperimentResult(
                name=exp_config["name"],
                type=exp_config["type"],
                config_path=str(config_path),
                success=False,
                error="Experiment timed out",
            )
        except Exception as e:
            return ExperimentResult(
                name=exp_config["name"],
                type=exp_config["type"],
                config_path=str(config_path),
                success=False,
                error=str(e),
            )
    
    def _run_convae_experiment(self, exp_config: dict[str, Any]) -> ExperimentResult:
        """Run a ConvAE experiment."""
        print(f"Running ConvAE experiment: {exp_config['name']}")
        
        # For now, ConvAE is not fully implemented
        # This would run the ConvAE training script when available
        
        return ExperimentResult(
            name=exp_config["name"],
            type=exp_config["type"],
            config_path=exp_config["config_path"],
            success=False,
            error="ConvAE experiments not yet implemented",
        )
    
    def _run_cra5_experiment(self, exp_config: dict[str, Any]) -> ExperimentResult:
        """Run a CRA5 adapter experiment."""
        print(f"Running CRA5 experiment: {exp_config['name']}")
        
        config_path = Path(exp_config["config_path"])
        if not config_path.exists():
            return ExperimentResult(
                name=exp_config["name"],
                type=exp_config["type"],
                config_path=str(config_path),
                success=False,
                error=f"Config file not found: {config_path}",
            )
        
        try:
            start_time = time.perf_counter()
            
            # Run CRA5 training script
            cmd = [
                "python", "scripts/train_cra5_adapter.py",
                "--config", str(config_path),
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config["runtime"]["timeout_minutes"] * 60,
            )
            
            runtime = time.perf_counter() - start_time
            
            if result.returncode != 0:
                return ExperimentResult(
                    name=exp_config["name"],
                    type=exp_config["type"],
                    config_path=str(config_path),
                    success=False,
                    error=f"CRA5 script failed: {result.stderr}",
                    runtime_seconds=runtime,
                )
            
            # Try to load metrics from output directory
            # Load the training config to get output directory
            with config_path.open() as f:
                train_config = yaml.safe_load(f)
            
            metrics_path = Path(train_config["experiment"]["output_dir"]) / "training_metrics.json"
            
            if metrics_path.exists():
                with metrics_path.open() as f:
                    metrics = json.load(f)
            else:
                metrics = {
                    "experiment_type": "cra5",
                    "config_path": str(config_path),
                    "success": True,
                    "stdout": result.stdout,
                }
            
            return ExperimentResult(
                name=exp_config["name"],
                type=exp_config["type"],
                config_path=str(config_path),
                success=True,
                metrics=metrics,
                runtime_seconds=runtime,
                output_dir=Path(train_config["experiment"]["output_dir"]),
            )
            
        except subprocess.TimeoutExpired:
            return ExperimentResult(
                name=exp_config["name"],
                type=exp_config["type"],
                config_path=str(config_path),
                success=False,
                error="Experiment timed out",
            )
        except Exception as e:
            return ExperimentResult(
                name=exp_config["name"],
                type=exp_config["type"],
                config_path=str(config_path),
                success=False,
                error=str(e),
            )
    
    def _run_single_experiment(self, exp_config: dict[str, Any]) -> ExperimentResult:
        """Run a single experiment based on its type."""
        if not exp_config.get("enabled", True):
            print(f"Skipping disabled experiment: {exp_config['name']}")
            return ExperimentResult(
                name=exp_config["name"],
                type=exp_config["type"],
                config_path=exp_config["config_path"],
                success=False,
                error="Experiment disabled",
            )
        
        exp_type = exp_config["type"]
        
        if exp_type == "pca":
            return self._run_pca_experiment(exp_config)
        elif exp_type == "convae":
            return self._run_convae_experiment(exp_config)
        elif exp_type == "cra5":
            return self._run_cra5_experiment(exp_config)
        else:
            return ExperimentResult(
                name=exp_config["name"],
                type=exp_config["type"],
                config_path=exp_config["config_path"],
                success=False,
                error=f"Unknown experiment type: {exp_type}",
            )
    
    def _check_dependencies(self, exp_config: dict[str, Any]) -> bool:
        """Check if experiment dependencies are satisfied."""
        depends_on = exp_config.get("depends_on", [])
        if not depends_on:
            return True
        
        # Check if all dependency experiments completed successfully
        completed_names = {r.name for r in self.results if r.success}
        
        for dep in depends_on:
            if dep not in completed_names:
                return False
        
        return True
    
    def _should_stop_early(self, current_result: ExperimentResult) -> bool:
        """Check if we should stop the ladder early."""
        stopping_config = self.config.get("stopping", {})
        min_improvement = stopping_config.get("min_improvement", 0.05)
        
        # Only apply early stopping to CRA5 experiments with sample sizes
        if current_result.type != "cra5" or not current_result.success:
            return False
        
        if not hasattr(current_result, "sample_size") or "sample_size" not in current_result.metrics:
            return False
        
        # Find previous CRA5 experiment with smaller sample size
        current_sample_size = current_result.metrics.get("subset_size", 0)
        
        prev_cra5_results = [
            r for r in self.results 
            if (r.type == "cra5" and r.success and 
                r.metrics and 
                r.metrics.get("subset_size", 0) < current_sample_size)
        ]
        
        if not prev_cra5_results:
            return False
        
        # Get the best previous result
        prev_best = min(
            prev_cra5_results,
            key=lambda r: r.metrics.get("best_val_loss", float("inf"))
        )
        
        # Check improvement
        current_val_loss = current_result.metrics.get("best_val_loss", float("inf"))
        prev_val_loss = prev_best.metrics.get("best_val_loss", float("inf"))
        
        if prev_val_loss == 0:
            return False
        
        improvement = (prev_val_loss - current_val_loss) / prev_val_loss
        
        print(f"Improvement check: {improvement:.3f} vs required {min_improvement:.3f}")
        
        if improvement < min_improvement:
            print(f"Early stopping: insufficient improvement ({improvement:.1%} < {min_improvement:.1%})")
            return True
        
        return False
    
    def _save_results(self) -> None:
        """Save ladder results to JSON."""
        # Prepare summary
        summary = {
            "ladder_config": self.config,
            "total_experiments": len(self.results),
            "successful_experiments": sum(1 for r in self.results if r.success),
            "failed_experiments": sum(1 for r in self.results if not r.success),
            "total_runtime_seconds": sum(r.runtime_seconds for r in self.results),
        }
        
        # Convert results to dictionaries
        results_data = []
        for result in self.results:
            result_dict = {
                "name": result.name,
                "type": result.type,
                "config_path": result.config_path,
                "success": result.success,
                "runtime_seconds": result.runtime_seconds,
            }
            
            if result.metrics:
                result_dict["metrics"] = result.metrics
            if result.error:
                result_dict["error"] = result.error
            if result.output_dir:
                result_dict["output_dir"] = str(result.output_dir)
            
            results_data.append(result_dict)
        
        # Save full results
        full_results = {
            "summary": summary,
            "results": results_data,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        results_path = self.output_dir / "experiment_ladder_results.json"
        with results_path.open("w") as f:
            json.dump(full_results, f, indent=2)
        
        print(f"Results saved to: {results_path}")
        
        # Save comparison table
        self._save_comparison_table()
    
    def _save_comparison_table(self) -> None:
        """Save a comparison table of successful experiments."""
        successful_results = [r for r in self.results if r.success and r.metrics]
        
        if not successful_results:
            print("No successful experiments to compare")
            return
        
        # Create comparison table
        table_rows = []
        
        for result in successful_results:
            row = {
                "name": result.name,
                "type": result.type,
                "runtime_seconds": result.runtime_seconds,
            }
            
            # Extract key metrics
            metrics = result.metrics
            if metrics:
                row.update({
                    "sample_size": metrics.get("subset_size", "N/A"),
                    "best_val_loss": metrics.get("best_val_loss", "N/A"),
                    "training_time": metrics.get("training_time_seconds", "N/A"),
                })
            
            table_rows.append(row)
        
        # Save table
        table_path = self.output_dir / "comparison_table.json"
        with table_path.open("w") as f:
            json.dump(table_rows, f, indent=2)
        
        print(f"Comparison table saved to: {table_path}")
        
        # Print summary table
        print("\n" + "="*80)
        print("EXPERIMENT LADDER SUMMARY")
        print("="*80)
        
        for row in table_rows:
            print(f"{row['name']:<20} {row['type']:<8} "
                  f"n={row['sample_size']:<4} "
                  f"val_loss={row['best_val_loss']:<10} "
                  f"time={row['runtime_seconds']:<8.1f}s")
    
    def run(self) -> None:
        """Run the complete experiment ladder."""
        experiments = self.config["experiments"]
        max_experiments = self.config.get("stopping", {}).get("max_experiments", len(experiments))
        
        print(f"Starting experiment ladder with {len(experiments)} experiments")
        print(f"Maximum experiments: {max_experiments}")
        
        for i, exp_config in enumerate(experiments):
            if len(self.results) >= max_experiments:
                print(f"Reached maximum experiments limit: {max_experiments}")
                break
            
            print(f"\n--- Experiment {i+1}/{len(experiments)}: {exp_config['name']} ---")
            
            # Check dependencies
            if not self._check_dependencies(exp_config):
                print(f"Skipping {exp_config['name']}: dependencies not satisfied")
                continue
            
            # Run experiment
            result = self._run_single_experiment(exp_config)
            self.results.append(result)
            
            # Print result
            if result.success:
                print(f"✅ {result.name} completed in {result.runtime_seconds:.1f}s")
            else:
                print(f"❌ {result.name} failed: {result.error}")
                
                if not self.config["runtime"]["continue_on_error"]:
                    print("Stopping ladder due to error")
                    break
            
            # Check early stopping
            if self._should_stop_early(result):
                print("Early stopping triggered")
                break
        
        # Save results
        print(f"\nLadder completed with {len(self.results)} experiments")
        self._save_results()


def main() -> None:
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run experiment ladder")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment_ladder.yaml"),
        help="Ladder configuration file",
    )
    
    args = parser.parse_args()
    
    if not args.config.exists():
        print(f"Config file not found: {args.config}")
        exit(1)
    
    # Run ladder
    ladder = ExperimentLadder(args.config)
    ladder.run()
    
    # Check if any experiments succeeded
    successful = sum(1 for r in ladder.results if r.success)
    total = len(ladder.results)
    
    print(f"\n🎉 Experiment ladder completed: {successful}/{total} experiments succeeded")
    
    if successful == 0:
        exit(1)


if __name__ == "__main__":
    main()