import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config.settings import EDA_FIGURES_DIR, EDA_RESULTS_DIR, RAFDB_LABEL_MAP


class EDAReport:
    def __init__(self, metadata: pd.DataFrame, zip_index: pd.DataFrame, label_table: pd.DataFrame):
        self.metadata = metadata
        self.zip_index = zip_index
        self.label_table = label_table
        sns.set_theme(style="whitegrid", palette="Set2")

    def save_all(self) -> dict:
        EDA_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        EDA_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        self.metadata.to_csv(EDA_RESULTS_DIR / "image_metadata.csv", index=False)
        self.zip_index.to_csv(EDA_RESULTS_DIR / "zip_index.csv", index=False)
        self.label_table.to_csv(EDA_RESULTS_DIR / "labels_detected.csv", index=False)
        self.metadata[self.metadata["is_duplicate"]].to_csv(
            EDA_RESULTS_DIR / "duplicate_images.csv", index=False
        )
        self.metadata[self.metadata["is_damaged"]].to_csv(
            EDA_RESULTS_DIR / "damaged_images.csv", index=False
        )
        self.metadata[self.metadata["label"].isna()].to_csv(
            EDA_RESULTS_DIR / "null_label_images.csv", index=False
        )
        self.metadata[self.metadata["label_mismatch"]].to_csv(
            EDA_RESULTS_DIR / "label_mismatches.csv", index=False
        )

        class_distribution = self._class_distribution()
        split_distribution = self._split_distribution()
        summary = self._summary(class_distribution, split_distribution)

        class_distribution.to_csv(EDA_RESULTS_DIR / "class_distribution.csv", index=False)
        split_distribution.to_csv(EDA_RESULTS_DIR / "split_distribution.csv", index=False)
        (EDA_RESULTS_DIR / "eda_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        self._plot_class_bar(class_distribution)
        self._plot_class_pie(class_distribution)
        self._plot_split_bar(split_distribution)
        self._plot_resolution_histograms()
        self._plot_examples_per_class()

        return summary

    def _class_distribution(self) -> pd.DataFrame:
        counts = (
            self.metadata.groupby(["label", "class_name"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values("label")
        )
        counts["percentage"] = 100 * counts["count"] / counts["count"].sum()
        return counts

    def _split_distribution(self) -> pd.DataFrame:
        counts = (
            self.metadata.groupby(["split", "label", "class_name"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["split", "label"])
        )
        counts["percentage_within_split"] = counts.groupby("split")["count"].transform(
            lambda s: 100 * s / s.sum()
        )
        return counts

    def _summary(self, class_distribution: pd.DataFrame, split_distribution: pd.DataFrame) -> dict:
        valid = self.metadata[~self.metadata["is_damaged"]].copy()
        widths = valid["width"].dropna()
        heights = valid["height"].dropna()
        formats = self.metadata["suffix"].value_counts().to_dict()
        split_counts = self.metadata["split"].value_counts().to_dict()
        label_counts = {
            f"{int(row.label)}_{row.class_name}": int(row["count"])
            for _, row in class_distribution.iterrows()
            if pd.notna(row.label)
        }

        max_count = int(class_distribution["count"].max())
        min_count = int(class_distribution["count"].min())
        imbalance_ratio = round(max_count / min_count, 4) if min_count else None

        return {
            "total_zip_entries": int(len(self.zip_index)),
            "total_images": int(len(self.metadata)),
            "train_images": int(split_counts.get("train", 0)),
            "test_images": int(split_counts.get("test", 0)),
            "unknown_split_images": int(split_counts.get("unknown", 0)),
            "num_classes": int(self.metadata["label"].nunique(dropna=True)),
            "classes": RAFDB_LABEL_MAP,
            "images_by_class": label_counts,
            "image_formats": formats,
            "resolution_average": {
                "width": round(float(widths.mean()), 2) if not widths.empty else None,
                "height": round(float(heights.mean()), 2) if not heights.empty else None,
            },
            "resolution_min": {
                "width": int(widths.min()) if not widths.empty else None,
                "height": int(heights.min()) if not heights.empty else None,
            },
            "resolution_max": {
                "width": int(widths.max()) if not widths.empty else None,
                "height": int(heights.max()) if not heights.empty else None,
            },
            "damaged_images": int(self.metadata["is_damaged"].sum()),
            "null_labels": int(self.metadata["label"].isna().sum()),
            "duplicate_images": int(self.metadata["is_duplicate"].sum()),
            "label_mismatches": int(self.metadata["label_mismatch"].sum()),
            "imbalance_ratio_max_min": imbalance_ratio,
            "is_imbalanced": bool(imbalance_ratio is not None and imbalance_ratio >= 2.0),
        }

    def _plot_class_bar(self, class_distribution: pd.DataFrame) -> None:
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=class_distribution, x="class_name", y="count", hue="class_name", legend=False)
        ax.set_title("Distribucion total por emocion")
        ax.set_xlabel("Emocion")
        ax.set_ylabel("Numero de imagenes")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(EDA_FIGURES_DIR / "class_distribution_bar.png", dpi=300)
        plt.close()

    def _plot_class_pie(self, class_distribution: pd.DataFrame) -> None:
        plt.figure(figsize=(8, 8))
        plt.pie(
            class_distribution["count"],
            labels=class_distribution["class_name"],
            autopct="%1.1f%%",
            startangle=90,
        )
        plt.title("Proporcion de clases RAF-DB")
        plt.tight_layout()
        plt.savefig(EDA_FIGURES_DIR / "class_distribution_pie.png", dpi=300)
        plt.close()

    def _plot_split_bar(self, split_distribution: pd.DataFrame) -> None:
        plt.figure(figsize=(11, 6))
        ax = sns.barplot(
            data=split_distribution,
            x="class_name",
            y="count",
            hue="split",
        )
        ax.set_title("Distribucion por split y emocion")
        ax.set_xlabel("Emocion")
        ax.set_ylabel("Numero de imagenes")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(EDA_FIGURES_DIR / "split_class_distribution_bar.png", dpi=300)
        plt.close()

    def _plot_resolution_histograms(self) -> None:
        valid = self.metadata[~self.metadata["is_damaged"]].copy()
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        sns.histplot(valid["width"], bins=30, ax=axes[0])
        axes[0].set_title("Histograma de anchos")
        sns.histplot(valid["height"], bins=30, ax=axes[1])
        axes[1].set_title("Histograma de altos")
        sns.histplot(valid["size_bytes"] / 1024, bins=30, ax=axes[2])
        axes[2].set_title("Histograma de tamanos (KB)")
        plt.tight_layout()
        plt.savefig(EDA_FIGURES_DIR / "resolution_and_size_histograms.png", dpi=300)
        plt.close()

    def _plot_examples_per_class(self, samples_per_class: int = 5) -> None:
        sample_df = (
            self.metadata[~self.metadata["is_damaged"]]
            .sort_values(["label", "path"])
            .groupby("label", group_keys=False)
            .head(samples_per_class)
        )
        labels = sorted(sample_df["label"].dropna().unique())
        if not labels:
            return

        fig, axes = plt.subplots(len(labels), samples_per_class, figsize=(samples_per_class * 2.2, len(labels) * 2.2))
        if len(labels) == 1:
            axes = [axes]

        for row_idx, label in enumerate(labels):
            rows = sample_df[sample_df["label"] == label].head(samples_per_class).reset_index(drop=True)
            for col_idx in range(samples_per_class):
                ax = axes[row_idx][col_idx] if len(labels) > 1 else axes[col_idx]
                ax.axis("off")
                if col_idx >= len(rows):
                    continue
                img = cv2.imread(rows.loc[col_idx, "path"])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img)
                if col_idx == 0:
                    ax.set_title(rows.loc[col_idx, "class_name"])
        plt.tight_layout()
        plt.savefig(EDA_FIGURES_DIR / "examples_per_class.png", dpi=300)
        plt.close()
