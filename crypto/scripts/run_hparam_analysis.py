#!/usr/bin/env python3
"""Master script to run complete hyperparameter tuning analysis pipeline."""
import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def run_script(script_name: str, description: str, **kwargs) -> bool:
    """Run a script and return success status."""
    logger.info(f"\n{'='*70}")
    logger.info(f"Step: {description}")
    logger.info(f"{'='*70}\n")

    cmd = [sys.executable, f"crypto/scripts/{script_name}"]

    for key, value in kwargs.items():
        cmd.extend([f"--{key.replace('_', '-')}", str(value)])

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        logger.info(f"\n✓ {description} completed successfully\n")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"\n✗ {description} failed with error code {e.returncode}\n")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run complete hyperparameter tuning analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script runs the entire analysis pipeline:
  1. Aggregate all model results into structured datasets
  2. Compare and rank models by various metrics
  3. Generate comprehensive visualizations
  4. Create detailed HTML/Markdown reports

Example usage:
  python crypto/scripts/run_hparam_analysis.py
  python crypto/scripts/run_hparam_analysis.py --hparam-dir results/hparam_tuning --top-n 20
        """,
    )

    parser.add_argument(
        "--hparam-dir",
        type=str,
        default="hparam_tuning",
        help="Directory containing hyperparameter tuning results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="hparam_tuning/analysis",
        help="Output directory for analysis results",
    )
    parser.add_argument(
        "--top-n", type=int, default=10, help="Number of top models to analyze"
    )
    parser.add_argument(
        "--skip-aggregation",
        action="store_true",
        help="Skip aggregation step (if already done)",
    )
    parser.add_argument(
        "--skip-comparison", action="store_true", help="Skip comparison step"
    )
    parser.add_argument(
        "--skip-visualization", action="store_true", help="Skip visualization step"
    )
    parser.add_argument(
        "--skip-report", action="store_true", help="Skip report generation step"
    )

    args = parser.parse_args()

    hparam_dir = Path(args.hparam_dir)
    output_dir = Path(args.output_dir)

    if not hparam_dir.exists():
        logger.error(f"Error: Hyperparameter directory not found: {hparam_dir}")
        return 1

    logger.info("\n" + "=" * 70)
    logger.info("HYPERPARAMETER TUNING ANALYSIS PIPELINE")
    logger.info("=" * 70)
    logger.info(f"\nInput directory: {hparam_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Top N models: {args.top_n}\n")

    results_file = output_dir / "all_results.pkl"
    rankings_dir = output_dir / "rankings"
    plots_dir = output_dir / "plots"

    steps_completed = 0
    steps_total = 4

    if not args.skip_aggregation:
        if run_script(
            "analyze_hparam_results.py",
            "1/4 - Aggregating Results",
            hparam_dir=args.hparam_dir,
            output_dir=args.output_dir,
        ):
            steps_completed += 1
        else:
            logger.error("Pipeline failed at aggregation step")
            return 1
    else:
        logger.info("Skipping aggregation step (using existing results)")
        steps_completed += 1

    if not results_file.exists():
        logger.error(f"Error: Results file not found: {results_file}")
        logger.error("Run without --skip-aggregation first")
        return 1

    if not args.skip_comparison:
        if run_script(
            "compare_hparam_models.py",
            "2/4 - Comparing and Ranking Models",
            results_file=str(results_file),
            output_dir=str(rankings_dir),
            top_n=args.top_n,
        ):
            steps_completed += 1
        else:
            logger.error("Pipeline failed at comparison step")
            return 1
    else:
        logger.info("Skipping comparison step")
        steps_completed += 1

    if not args.skip_visualization:
        if run_script(
            "visualize_hparam_results.py",
            "3/4 - Generating Visualizations",
            results_file=str(results_file),
            output_dir=str(plots_dir),
            top_n=args.top_n,
        ):
            steps_completed += 1
        else:
            logger.error("Pipeline failed at visualization step")
            return 1
    else:
        logger.info("Skipping visualization step")
        steps_completed += 1

    if not args.skip_report:
        if run_script(
            "generate_hparam_report.py",
            "4/4 - Generating Comprehensive Report",
            results_file=str(results_file),
            rankings_dir=str(rankings_dir),
            plots_dir=str(plots_dir),
            output_dir=str(output_dir),
        ):
            steps_completed += 1
        else:
            logger.error("Pipeline failed at report generation step")
            return 1
    else:
        logger.info("Skipping report generation step")
        steps_completed += 1

    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)
    logger.info(f"\nSteps completed: {steps_completed}/{steps_total}\n")

    logger.info("Generated files:")
    logger.info(f"  - Data: {results_file}")
    logger.info(f"  - Rankings: {rankings_dir}/")
    logger.info(f"  - Plots: {plots_dir}/")
    logger.info(f"  - Report (HTML): {output_dir}/report.html")
    logger.info(f"  - Report (MD): {output_dir}/report.md")

    logger.info(f"\nTo view the HTML report:")
    logger.info(f"  open {output_dir}/report.html")

    logger.info(f"\nTo view the CSV data:")
    logger.info(f"  open {output_dir}/all_results.csv")

    return 0


if __name__ == "__main__":
    sys.exit(main())
