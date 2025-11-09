#!/usr/bin/env python3
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def generate_markdown_report(
    df: pd.DataFrame, rankings_dir: Path, plots_dir: Path
) -> str:
    """Generate comprehensive Markdown report."""
    report = []

    report.append("# Hyperparameter Tuning Analysis Report")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    report.append("## Executive Summary\n")
    report.append(f"- **Total Models Analyzed**: {len(df)} (actual hedgers only)")
    report.append(
        f"- **Note**: Models with variance ratio > 2.0 were filtered out (not hedging)"
    )
    report.append(
        f"- **Model Types**: {', '.join([f'{k}: {v}' for k, v in df['model_type'].value_counts().to_dict().items()])}"
    )
    report.append(f"- **Layers Tested**: {sorted(df['n_layers'].unique())}")
    report.append(f"- **Units Tested**: {sorted(df['n_units'].unique())}")
    report.append(
        f"- **Risk Measures**: {', '.join([f'{k}: {v}' for k, v in df['risk_measure'].value_counts().to_dict().items()])}"
    )

    best_model = df.nlargest(1, "dh_sharpe_ratio").iloc[0]
    report.append(f"\n### Best Model (by Sharpe Ratio)")
    report.append(f"- **ID**: `{best_model['model_dir']}`")
    report.append(
        f"- **Architecture**: {best_model['model_type'].upper()}, {best_model['n_layers']} layers, {best_model['n_units']} units"
    )
    report.append(
        f"- **Risk Measure**: {best_model['risk_measure']} ({best_model['risk_param']})"
    )
    report.append(f"- **Learning Rate**: {best_model['learning_rate']}")
    report.append(f"- **Sharpe Ratio**: {best_model['dh_sharpe_ratio']:.4f}")
    report.append(f"- **Mean PnL**: ${best_model['dh_mean']:.2f}")
    report.append(f"- **CVaR 95%**: ${best_model['dh_cvar_95']:.2f}")
    report.append(f"- **Win Rate**: {best_model['dh_win_rate']:.1%}")
    report.append(
        f"- **Variance Ratio**: {best_model['variance_ratio']:.2f}× (vs Black-Scholes)"
    )

    report.append("\n## Performance Overview\n")

    report.append("### Key Statistics")
    report.append("\n| Metric | Mean | Std | Min | Max |")
    report.append("|--------|------|-----|-----|-----|")
    for metric in ["dh_sharpe_ratio", "dh_mean", "dh_cvar_95", "dh_win_rate"]:
        stats = df[metric].agg(["mean", "std", "min", "max"])
        report.append(
            f"| {metric.replace('dh_', '').replace('_', ' ').title()} | "
            f"{stats['mean']:.4f} | {stats['std']:.4f} | "
            f"{stats['min']:.4f} | {stats['max']:.4f} |"
        )

    report.append("\n### Distribution Visualization")
    if (plots_dir / "metric_distributions.png").exists():
        report.append("\n![Metric Distributions](plots/metric_distributions.png)\n")

    report.append("## Architecture Comparison\n")

    arch_perf = df.groupby("model_type")[
        ["dh_sharpe_ratio", "dh_mean", "dh_cvar_95", "dh_win_rate"]
    ].mean()
    report.append("### Average Performance by Architecture")
    report.append("\n| Architecture | Sharpe Ratio | Mean PnL | CVaR 95% | Win Rate |")
    report.append("|--------------|--------------|----------|----------|----------|")
    for model_type, row in arch_perf.iterrows():
        report.append(
            f"| {model_type.upper()} | {row['dh_sharpe_ratio']:.4f} | "
            f"${row['dh_mean']:.2f} | ${row['dh_cvar_95']:.2f} | {row['dh_win_rate']:.1%} |"
        )

    if (plots_dir / "performance_by_architecture.png").exists():
        report.append(
            "\n![Performance by Architecture](plots/performance_by_architecture.png)\n"
        )

    report.append("## Hyperparameter Analysis\n")

    report.append("### Impact of Number of Layers")
    layers_impact = df.groupby("n_layers")["dh_sharpe_ratio"].agg(
        ["mean", "std", "count"]
    )
    report.append("\n| Layers | Mean Sharpe | Std | Count |")
    report.append("|--------|-------------|-----|-------|")
    for layers, row in layers_impact.iterrows():
        report.append(
            f"| {layers} | {row['mean']:.4f} | {row['std']:.4f} | {int(row['count'])} |"
        )

    report.append("\n### Impact of Number of Units")
    units_impact = df.groupby("n_units")["dh_sharpe_ratio"].agg(
        ["mean", "std", "count"]
    )
    report.append("\n| Units | Mean Sharpe | Std | Count |")
    report.append("|-------|-------------|-----|-------|")
    for units, row in units_impact.iterrows():
        report.append(
            f"| {units} | {row['mean']:.4f} | {row['std']:.4f} | {int(row['count'])} |"
        )

    report.append("\n### Impact of Risk Measure")
    risk_impact = df.groupby("risk_measure")["dh_sharpe_ratio"].agg(
        ["mean", "std", "count"]
    )
    report.append("\n| Risk Measure | Mean Sharpe | Std | Count |")
    report.append("|--------------|-------------|-----|-------|")
    for risk_measure, row in risk_impact.iterrows():
        report.append(
            f"| {risk_measure} | {row['mean']:.4f} | {row['std']:.4f} | {int(row['count'])} |"
        )

    if (plots_dir / "heatmap_n_layers_n_units_dh_sharpe_ratio.png").exists():
        report.append(
            "\n![Heatmap: Layers vs Units](plots/heatmap_n_layers_n_units_dh_sharpe_ratio.png)\n"
        )

    if (plots_dir / "risk_measure_comparison.png").exists():
        report.append("![Risk Measure Comparison](plots/risk_measure_comparison.png)\n")

    report.append("## Deep Hedge vs Black-Scholes\n")

    beats_sharpe = (df["dh_sharpe_ratio"] > df["bs_sharpe_ratio"]).sum()
    beats_mean = (df["dh_mean"] > df["bs_mean"]).sum()
    beats_cvar = (df["dh_cvar_95"] > df["bs_cvar_95"]).sum()

    report.append(
        f"- **Models beating BS Sharpe**: {beats_sharpe}/{len(df)} ({100*beats_sharpe/len(df):.1f}%)"
    )
    report.append(
        f"- **Models beating BS Mean PnL**: {beats_mean}/{len(df)} ({100*beats_mean/len(df):.1f}%)"
    )
    report.append(
        f"- **Models beating BS CVaR**: {beats_cvar}/{len(df)} ({100*beats_cvar/len(df):.1f}%)"
    )

    avg_sharpe_improvement = df["dh_sharpe_improvement"].mean()
    avg_mean_improvement = df["dh_mean_improvement"].mean()
    avg_cvar_improvement = df["dh_cvar_improvement"].mean()

    report.append(f"\n### Average Improvements")
    report.append(f"- **Sharpe Ratio**: {avg_sharpe_improvement:+.4f}")
    report.append(f"- **Mean PnL**: ${avg_mean_improvement:+.2f}")
    report.append(f"- **CVaR 95%**: ${avg_cvar_improvement:+.2f}")

    if (plots_dir / "deep_vs_baseline.png").exists():
        report.append("\n![Deep Hedge vs Baseline](plots/deep_vs_baseline.png)\n")

    report.append("## Top Performers\n")

    report.append("### Top 10 Models by Sharpe Ratio")
    top10 = df.nlargest(10, "dh_sharpe_ratio")
    report.append(
        "\n| Rank | Model | Type | Layers | Units | Risk | Sharpe | Mean PnL | Win Rate |"
    )
    report.append(
        "|------|-------|------|--------|-------|------|--------|----------|----------|"
    )
    for rank, (_, row) in enumerate(top10.iterrows(), 1):
        report.append(
            f"| {rank} | `{row['model_dir'][:16]}...` | {row['model_type'].upper()} | "
            f"{row['n_layers']} | {row['n_units']} | {row['risk_measure'][:4]} | "
            f"{row['dh_sharpe_ratio']:.4f} | ${row['dh_mean']:.0f} | {row['dh_win_rate']:.1%} |"
        )

    if (plots_dir / "top_10_models_comparison.png").exists():
        report.append(
            "\n![Top 10 Models Comparison](plots/top_10_models_comparison.png)\n"
        )

    report.append("## Training Insights\n")

    report.append("### Training Loss Improvement")
    avg_improvement = df["train_improvement_pct"].mean()
    report.append(f"- **Average training loss improvement**: {avg_improvement:.2f}%")

    high_improvement = df[df["train_improvement_pct"] > 15]
    report.append(
        f"- **Models with >15% improvement**: {len(high_improvement)} ({100*len(high_improvement)/len(df):.1f}%)"
    )

    correlation = df[["train_improvement_pct", "dh_sharpe_ratio"]].corr().iloc[0, 1]
    report.append(
        f"- **Correlation (training improvement vs backtest Sharpe)**: {correlation:.3f}"
    )

    if (plots_dir / "correlation_matrix.png").exists():
        report.append("\n![Correlation Matrix](plots/correlation_matrix.png)\n")

    report.append("## Risk-Return Analysis\n")

    if (plots_dir / "risk_return_tradeoff.png").exists():
        report.append("![Risk-Return Tradeoff](plots/risk_return_tradeoff.png)\n")

    report.append("## Key Findings & Recommendations\n")

    report.append("### Winning Patterns")

    best_architecture = arch_perf["dh_sharpe_ratio"].idxmax()
    report.append(
        f"1. **Best Architecture**: {best_architecture.upper()} models performed best on average"
    )

    best_layers = layers_impact["mean"].idxmax()
    report.append(
        f"2. **Optimal Depth**: {best_layers} layers showed best average performance"
    )

    best_units = units_impact["mean"].idxmax()
    report.append(f"3. **Optimal Width**: {best_units} units demonstrated best results")

    best_risk = risk_impact["mean"].idxmax()
    report.append(
        f"4. **Risk Measure**: {best_risk} risk measure outperformed alternatives"
    )

    report.append("\n### Model Selection Guidance")
    report.append(f"- **For production deployment**: Use `{best_model['model_dir']}`")
    report.append(f"  - Highest Sharpe ratio: {best_model['dh_sharpe_ratio']:.4f}")
    report.append(f"  - Improvement over BS: {best_model['dh_sharpe_improvement']:.4f}")

    conservative_models = df[df["dh_cvar_95"] > df["dh_cvar_95"].quantile(0.9)]
    if len(conservative_models) > 0:
        best_conservative = conservative_models.nlargest(1, "dh_sharpe_ratio").iloc[0]
        report.append(
            f"\n- **For risk-averse strategies**: Use `{best_conservative['model_dir']}`"
        )
        report.append(f"  - Sharpe ratio: {best_conservative['dh_sharpe_ratio']:.4f}")
        report.append(f"  - CVaR 95%: ${best_conservative['dh_cvar_95']:.2f} (top 10%)")

    report.append("\n### Areas for Improvement")

    if beats_cvar < 0.1 * len(df):
        report.append(
            "- **Tail Risk**: Only {:.1%} of models beat BS on CVaR - focus on tail risk optimization".format(
                beats_cvar / len(df)
            )
        )

    if df["dh_win_rate"].max() < 0.5:
        report.append(
            f"- **Win Rate**: Max win rate is {df['dh_win_rate'].max():.1%} - consider alternative strategies for consistency"
        )

    if correlation < 0.3:
        report.append(
            f"- **Overfitting Risk**: Low correlation ({correlation:.3f}) between training improvement and backtest performance"
        )

    report.append("\n---")
    report.append("\n*Report generated by automated hyperparameter analysis pipeline*")

    return "\n".join(report)


def markdown_to_html(markdown_content: str) -> str:
    """Convert markdown to simple HTML."""
    html = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        '<meta charset="UTF-8">',
        "<title>Hyperparameter Tuning Report</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }",
        "h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }",
        "h2 { color: #34495e; border-bottom: 2px solid #95a5a6; padding-bottom: 8px; margin-top: 30px; }",
        "h3 { color: #7f8c8d; margin-top: 20px; }",
        "table { border-collapse: collapse; width: 100%; margin: 15px 0; }",
        "th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }",
        "th { background-color: #3498db; color: white; }",
        "tr:nth-child(even) { background-color: #f2f2f2; }",
        "img { max-width: 100%; height: auto; margin: 20px 0; }",
        "code { background-color: #f4f4f4; padding: 2px 6px; border-radius: 3px; font-family: monospace; }",
        "ul, ol { line-height: 1.8; }",
        "</style>",
        "</head>",
        "<body>",
    ]

    lines = markdown_content.split("\n")
    in_table = False

    for line in lines:
        if line.startswith("# "):
            html.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            html.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("### "):
            html.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("|") and "|" in line:
            if not in_table:
                html.append("<table>")
                in_table = True
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if all(set(c) <= {"-", " ", ":"} for c in cells):
                continue
            tag = (
                "th"
                if cells[0]
                and not any(
                    c.replace(".", "")
                    .replace("-", "")
                    .replace("$", "")
                    .replace("%", "")
                    .replace(",", "")
                    .strip()
                    .replace(" ", "")
                    .isdigit()
                    for c in cells
                )
                else "td"
            )
            html.append(
                "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>"
            )
        elif in_table and not line.startswith("|"):
            html.append("</table>")
            in_table = False
        elif line.startswith("!["):
            alt_end = line.index("]")
            url_start = line.index("(", alt_end) + 1
            url_end = line.index(")", url_start)
            alt = line[2:alt_end]
            url = line[url_start:url_end]
            html.append(f'<img src="{url}" alt="{alt}">')
        elif line.startswith("- "):
            if not html[-1].startswith("<li>") and not html[-1] == "<ul>":
                html.append("<ul>")
            html.append(f"<li>{line[2:]}</li>")
        elif html[-1].startswith("<li>") and not line.startswith("- "):
            html.append("</ul>")
            if line.strip():
                html.append(f"<p>{line}</p>")
        elif line.strip():
            line = line.replace("`", "<code>").replace("`", "</code>")
            html.append(f"<p>{line}</p>")
        elif html[-1] != "<br>":
            html.append("<br>")

    if in_table:
        html.append("</table>")
    if html[-1].startswith("<li>"):
        html.append("</ul>")

    html.extend(["</body>", "</html>"])
    return "\n".join(html)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate comprehensive hyperparameter tuning report"
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default="hparam_tuning/analysis/all_results.pkl",
        help="Path to aggregated results file",
    )
    parser.add_argument(
        "--rankings-dir",
        type=str,
        default="hparam_tuning/analysis/rankings",
        help="Directory containing ranking files",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="hparam_tuning/analysis/plots",
        help="Directory containing plot files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="hparam_tuning/analysis",
        help="Output directory for report",
    )
    args = parser.parse_args()

    results_file = Path(args.results_file)
    rankings_dir = Path(args.rankings_dir)
    plots_dir = Path(args.plots_dir)
    output_dir = Path(args.output_dir)

    logger.info(f"Loading results from {results_file}")
    df = pd.read_pickle(results_file)
    logger.info(f"Loaded {len(df)} models")

    logger.info("\nGenerating comprehensive report...")
    markdown_content = generate_markdown_report(df, rankings_dir, plots_dir)

    md_file = output_dir / "report.md"
    with open(md_file, "w") as f:
        f.write(markdown_content)
    logger.info(f"Saved Markdown report to {md_file}")

    logger.info("\nConverting to HTML...")
    html_content = markdown_to_html(markdown_content)

    html_file = output_dir / "report.html"
    with open(html_file, "w") as f:
        f.write(html_content)
    logger.info(f"Saved HTML report to {html_file}")

    logger.info(f"\nReport generation complete!")
    logger.info(f"  Markdown: {md_file}")
    logger.info(f"  HTML: {html_file}")
    logger.info(f"\nTo view the HTML report, run: open {html_file}")


if __name__ == "__main__":
    main()
