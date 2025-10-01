import shutil
from pathlib import Path
import run
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import json
import textwrap
from PIL import Image

#Enter the name of your test 
TEST_NAME = "Individual_controls_testing"

#Enter the relative path of your test case json
TEST_JSON_PATH = r"test_cases\combination_of_controls.json"

'''--------------------------------------------------------------------------------------------------------------------------------'''

def get_folder_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.glob("**/*") if f.is_file())

# ==============================
# Load test cases from JSON
# ==============================
def load_test_cases(json_file=Path(TEST_JSON_PATH)):
    with open(json_file, "r", encoding="utf-8") as f:
        raw_cases = json.load(f)

    # Convert SAMPLE_INPUT_DIR strings into Path objects
    for case in raw_cases:
        case["SAMPLE_INPUT_DIR"] = Path(case["SAMPLE_INPUT_DIR"])
        case["SAMPLE_OUTPUT_DIR"] = Path("test_sample")  # always reset to same folder
    return raw_cases


def run_experiments(test_cases, run_dir: Path):
    results = []
    buffer_dir = run_dir / "buffers"
    buffer_dir.mkdir(parents=True, exist_ok=True)

    for idx, case in enumerate(test_cases):
        print(f"\n=== Running test case: {case['name']} ===")

        # Reset final results dir before each case
        if run.FINAL_RESULT_DIR.exists():
            shutil.rmtree(run.FINAL_RESULT_DIR)
        run.FINAL_RESULT_DIR.mkdir(parents=True, exist_ok=True)

        # Merge with defaults
        merged_cfg = {**run.CONFIG, **case}

        # Run pipeline with this test case config
        run.pipeline(merged_cfg)

        # Measure sizes
        sample_size = get_folder_size(merged_cfg["SAMPLE_OUTPUT_DIR"])
        final_size = get_folder_size(run.FINAL_RESULT_DIR)
        compression = 100 * (1 - final_size / sample_size) if sample_size > 0 else 0

        # ---- Pick one sample + corresponding result to buffer ----
        sample_images = list(merged_cfg["SAMPLE_OUTPUT_DIR"].glob("*.*"))
        final_images = list(run.FINAL_RESULT_DIR.glob("*.*"))

        if sample_images and final_images:
            sample_img = sample_images[0]
            final_img = final_images[0]

            case_buf_dir = buffer_dir / f"case_{idx+1}"
            case_buf_dir.mkdir(parents=True, exist_ok=True)

            # Copy
            sample_copy = case_buf_dir / f"sample{sample_img.suffix}"
            final_copy = case_buf_dir / f"final{final_img.suffix}"
            shutil.copy(sample_img, sample_copy)
            shutil.copy(final_img, final_copy)

        # Record result
        results.append({
            "Case": merged_cfg["name"],
            "Format": merged_cfg["OUTPUT_FORMAT"],
            "JPEG_Q": merged_cfg["JPEG_QUALITY"],
            "WEBP_Q": merged_cfg["WEBP_QUALITY"],
            "Scale": merged_cfg["SCALE_FACTOR"],
            "ConvexHull": merged_cfg["TOGGLE_CONVEX_HULL"],
            "Crop": merged_cfg["TOGGLE_CONVEX_HULL_CROP"],
            "Grayscale": merged_cfg["TOGGLE_GRAYSCALE"],
            "Downscale": merged_cfg["TOGGLE_DOWNSCALE"],
            "SampleSizeKB": sample_size / 1024,
            "FinalSizeKB": final_size / 1024,
            "Compression%": compression,
        })

    return pd.DataFrame(results)


def generate_report(df: pd.DataFrame, run_dir: Path):
    # Wrap long text in Case column for better fit
    df = df.copy()
    df["Case"] = df["Case"].apply(lambda x: "\n".join(textwrap.wrap(x, width=18)))

    # Save CSV
    df.to_csv(run_dir / "compression_report.csv", index=False)

    # Bar chart
    plt.figure(figsize=(10,6))
    bar_width = 0.35
    x = range(len(df))
    plt.bar([i - bar_width/2 for i in x], df["SampleSizeKB"], width=bar_width, label="Test Sample (total)")
    plt.bar([i + bar_width/2 for i in x], df["FinalSizeKB"], width=bar_width, label="Final Result (total)")
    plt.xticks(x, df["Case"], rotation=20, ha="right")
    plt.ylabel("Size (KB)")
    plt.title("Total Folder Size: Test Sample vs Final Result")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "sizes_comparison.png")

    # Line chart
    plt.figure(figsize=(10,6))
    plt.plot(df["Case"], df["Compression%"], marker="o")
    plt.ylabel("Compression (%)")
    plt.title("Compression Achieved per Test Case")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(run_dir / "compression_percent.png")

    # Combine into PDF
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(run_dir / "compression_report.pdf") as pdf:
        # --- Fix table layout ---
        # Drop only per-image sizes (if present)
        df_table = df.drop(columns=["SampleImageSizeKB", "FinalImageSizeKB"], errors="ignore")

        fig, ax = plt.subplots(figsize=(12,4))
        ax.axis("tight"); ax.axis("off")
        table = ax.table(cellText=df_table.round(2).values,
                         colLabels=df_table.columns,
                         loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.4)
        table.auto_set_column_width(col=list(range(len(df_table.columns))))

        # --- Dynamic row heights ---
        for (row, col), cell in table.get_celld().items():
            if row == 0:  # header
                cell.set_height(0.12)
                cell.set_text_props(weight="bold", ha="center", va="center")
            else:
                txt = str(df_table.iloc[row-1, col])
                num_lines = txt.count("\n") + 1
                cell.set_height(0.08 * num_lines)
                cell.set_text_props(ha="center", va="center")

        pdf.savefig(fig)

        # Add graphs
        pdf.savefig(plt.figure(1))
        pdf.savefig(plt.figure(2))

        # ---- Add case-by-case image slides ----
        buffer_dir = run_dir / "buffers"
        for idx, case in enumerate(df["Case"], start=1):
            case_buf_dir = buffer_dir / f"case_{idx}"
            sample_candidates = list(case_buf_dir.glob("sample.*"))
            final_candidates = list(case_buf_dir.glob("final.*"))
            if not sample_candidates or not final_candidates:
                continue

            sample_img = sample_candidates[0]
            final_img = final_candidates[0]

            # Get resolution
            s_res = Image.open(sample_img).size
            f_res = Image.open(final_img).size

            # Get sizes
            s_size = sample_img.stat().st_size / 1024
            f_size = final_img.stat().st_size / 1024

            # Get quality info
            row = df.iloc[idx-1]
            q_info = f"JPEG_Q={row['JPEG_Q']} | WEBP_Q={row['WEBP_Q']} | Format={row['Format']}"

            # Plot side-by-side with resolution + size
            fig, axs = plt.subplots(1, 2, figsize=(12,6))
            for ax, img_path, title, res, size_kb in [
                (axs[0], sample_img, "Sample", s_res, s_size),
                (axs[1], final_img, "Final", f_res, f_size),
            ]:
                img = plt.imread(img_path)
                ax.imshow(img)
                ax.set_title(f"{title}\n{res[0]}x{res[1]} | {size_kb:.1f} KB")
                if title == "Final":
                    ax.set_xlabel(q_info)
                ax.set_xticks(range(0, res[0], max(1,res[0]//5)))
                ax.set_yticks(range(0, res[1], max(1,res[1]//5)))

            plt.suptitle(f"Case {idx}: {row['Case']}")
            pdf.savefig(fig)
            plt.close(fig)

    print(f"✅ Report generated in: {run_dir}")


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("test_cases") / TEST_NAME / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    cases = load_test_cases(TEST_JSON_PATH)
    df = run_experiments(cases, run_dir)
    generate_report(df, run_dir)
