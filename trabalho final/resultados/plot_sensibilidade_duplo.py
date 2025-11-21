from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import re
import unicodedata

DATA_PATH = Path(__file__).resolve().parent / "AnaliseSensibilidade.xlsx"
SHEET_NAME = "Dados"
OUTPUT_DIR = Path(__file__).resolve().parent / "figuras"
GENERATE_COLOR = True
GENERATE_MONO = True
COLOR_CYCLE = plt.rcParams["axes.prop_cycle"].by_key()["color"]
MARKERS = ["o", "s", "^", "v", "D", "P", "X", "*"]


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_str = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_str.lower()).strip("_")
    return slug or "figura"


def plot_metric(df, x_col, column_template, y_label, title, filename, mono=False, cases=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = df[x_col]
    cases = list(cases) if cases is not None else list(range(1, 9))
    for idx, case_idx in enumerate(cases):
        col = column_template.format(case_idx)
        if mono:
            marker = MARKERS[idx % len(MARKERS)]
            ax.plot(
                x,
                df[col],
                color="black",
                linewidth=1,
                marker=marker,
                markersize=3,
                label=f"Caso {case_idx}",
            )
        else:
            color = COLOR_CYCLE[idx % len(COLOR_CYCLE)]
            ax.plot(x, df[col], linewidth=1, color=color, label=f"Caso {case_idx}")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    filepath = Path(filename)
    if not filepath.is_absolute():
        filepath = OUTPUT_DIR / filepath
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=300)
    plt.close(fig)
    fig.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
    x_col = "Carga térmica refervedor [Gcal/h]"
    metrics = [
        {
            "column_template": "T fundo [°C] - Caso {}",
            "y_label": "Temperatura de fundo [°C]",
            "title": "Carga térmica refervedor x T Fundo",
            "base_filename": "q_ref_vs_t_fundo",
        },
        {
            "column_template": "T topo [°C] - Caso {}",
            "y_label": "Temperatura de topo [°C]",
            "title": "Carga térmica  refervedor x T Topo",
            "base_filename": "q_ref_vs_t_topo",
        },
        {
            "column_template": "Recuperação H2S [%] - Caso {}",
            "y_label": "Recuperação de H2S [%]",
            "title": "Carga térmica refervedor x Recuperação H2S",
            "base_filename": "q_ref_vs_recuperacao_h2s",
        },
        {
            "column_template": "Perda NH3 [%] - Caso {}",
            "y_label": "Perda de NH3 [%]",
            "title": "Carga térmica refervedor x Perda NH3",
            "base_filename": "q_ref_vs_perda_nh3",
        },
    ]
    case_groups = [
        {"cases": [1, 2, 3, 4], "label": "influência contaminantes"},
        {"cases": [1, 5, 6], "label": "influência vazão carga"},
        {"cases": [1, 7, 8], "label": "influência temperatura carga"},
    ]

    for metric in metrics:
        base = metric["base_filename"]
        if GENERATE_COLOR:
            plot_metric(
                df,
                x_col,
                metric["column_template"],
                metric["y_label"],
                metric["title"],
                f"{base}.png",
                mono=False,
            )
        if GENERATE_MONO:
            plot_metric(
                df,
                x_col,
                metric["column_template"],
                metric["y_label"],
                metric["title"],
                f"{base}_mono.png",
                mono=True,
            )
        for group in case_groups:
            suffix = slugify(group["label"])
            title = f"{metric['title']} - {group['label']}"
            group_dir = OUTPUT_DIR / suffix
            group_dir.mkdir(parents=True, exist_ok=True)
            if GENERATE_COLOR:
                filename_color = group_dir / f"{base}_{suffix}.png"
                plot_metric(
                    df,
                    x_col,
                    metric["column_template"],
                    metric["y_label"],
                    title,
                    filename_color.name,
                    mono=False,
                    cases=group["cases"],
                )
            if GENERATE_MONO:
                filename_mono = group_dir / f"{base}_{suffix}_mono.png"
                plot_metric(
                    df,
                    x_col,
                    metric["column_template"],
                    metric["y_label"],
                    title,
                    filename_mono.name,
                    mono=True,
                    cases=group["cases"],
                )
            filename_mono = f"{base}_{suffix}_mono.png"
            plot_metric(
                df,
                x_col,
                metric["column_template"],
                metric["y_label"],
                title,
                filename_color,
                mono=False,
                cases=group["cases"],
            )
            plot_metric(
                df,
                x_col,
                metric["column_template"],
                metric["y_label"],
                title,
                filename_mono,
                mono=True,
                cases=group["cases"],
            )


if __name__ == "__main__":
    main()
