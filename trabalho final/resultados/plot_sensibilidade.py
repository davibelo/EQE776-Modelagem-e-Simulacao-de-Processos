from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA_PATH = Path("AnaliseSensibilidade.xlsx")
SHEET_NAME = "Dados"
DATA_PATH = Path(__file__).resolve().parent / "AnaliseSensibilidade.xlsx"
OUTPUT_DIR = Path(__file__).resolve().parent / "figuras"


def plot_metric(df, x_col, metric_name, column_template, y_label, title, filename):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = df[x_col]

    for case_idx in range(1, 7):
        col = column_template.format(case_idx)
        ax.plot(x, df[col], marker="o", linewidth=1.5, label=f"Caso {case_idx}")

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
    x_col = "Q refervedor [Gcal/h]"

    metrics = [
        {
            "metric_name": "T fundo [°C]",
            "column_template": "T fundo [°C] - Caso {}",
            "y_label": "Temperatura de fundo [°C]",
            "title": "Q refervedor x T Fundo",
            "filename": "q_ref_vs_t_fundo.png",
        },
        {
            "metric_name": "T topo [°C]",
            "column_template": "T topo [°C] - Caso {}",
            "y_label": "Temperatura de topo [°C]",
            "title": "Q refervedor x T Topo",
            "filename": "q_ref_vs_t_topo.png",
        },
        {
            "metric_name": "Recuperação H2S [%]",
            "column_template": "Recuperação H2S [%] - Caso {}",
            "y_label": "Recuperação de H2S [%]",
            "title": "Q refervedor x Recuperação H2S",
            "filename": "q_ref_vs_recuperacao_h2s.png",
        },
        {
            "metric_name": "Perda NH3 [%]",
            "column_template": "Perda NH3 [%] - Caso {}",
            "y_label": "Perda de NH3 [%]",
            "title": "Q refervedor x Perda NH3",
            "filename": "q_ref_vs_perda_nh3.png",
        },
    ]

    for metric in metrics:
        plot_metric(
            df,
            x_col,
            metric["metric_name"],
            metric["column_template"],
            metric["y_label"],
            metric["title"],
            metric["filename"],
        )


if __name__ == "__main__":
    main()
