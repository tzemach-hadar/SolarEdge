# --- file: validation_exporter.py ---
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# פרויקטי אימפורט (עדכני נתיבים לפי הפרויקט שלך)
from training.training_module import BertLitModule
from training.dataset_module import ChatDataset
# מיפויים מתוך data_labeling.py (דאגי שהם יוגדרו שם כמשתנים גלובליים)
from data_preprocessing.data_labeling import product_map, issue_map


def invert_map(m: dict) -> dict:
    """Invert a mapping dict[str->int] -> dict[int->str]."""
    return {v: k for k, v in m.items()}


class ValidationExporter:
    """
    Run validation predictions on test set, merge with labeled metadata by ID,
    and export a CSV with all original fields + predictions.

    Assumes the classifier predicts product labels (indices compatible with product_map).
    """

    def __init__(
        self,
        model_ckpt_path: str,
        test_pt_path: str,
        labeled_csv_path: str,
        output_csv_path: str = "validation_full_test_with_preds.csv",
        id_col_meta: str = "chat_id",
        batch_size: int = 16,
        save_confusion_matrix: bool = True,
    ):
        self.model_ckpt_path = model_ckpt_path
        self.test_pt_path = test_pt_path
        self.labeled_csv_path = labeled_csv_path
        self.output_csv_path = output_csv_path
        self.id_col_meta = id_col_meta
        self.batch_size = batch_size
        self.save_confusion_matrix = save_confusion_matrix

        # Maps from data_labeling.py
        self.product_map = product_map
        self.issue_map = issue_map
        self.idx2product = invert_map(self.product_map)
        self.idx2issue = invert_map(self.issue_map)

        # Runtime holders
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.test_dataset = None

    # -----------------------------
    # Loading
    # -----------------------------
    def load_model(self):
        self.model = BertLitModule.load_from_checkpoint(self.model_ckpt_path)
        self.model.eval().to(self.device)

    def load_test_dataset(self):
        self.test_dataset = ChatDataset(self.test_pt_path)

    # -----------------------------
    # Inference on test
    # -----------------------------
    def _run_predictions(self):
        """
        Iterate over test loader; return list of rows with id/true/pred/conf + y_true/y_pred for metrics.
        Expects ChatDataset to return a dict with keys: input_ids, attention_mask, labels, (optional) ids
        """
        loader = DataLoader(self.test_dataset, batch_size=self.batch_size)
        rows, y_true, y_pred = [], [], []

        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            ids = batch.get("ids", None)

            with torch.no_grad():
                logits = self.model.model(input_ids, attention_mask)  # [B, C]
                probs = torch.softmax(logits, dim=1)                  # [B, C]
                preds = torch.argmax(probs, dim=1)                    # [B]

            # collect metrics arrays
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(preds.detach().cpu().tolist())

            # build rows
            for i in range(preds.shape[0]):
                true_idx = int(labels[i].detach().cpu().item())
                pred_idx = int(preds[i].detach().cpu().item())
                conf = float(probs[i, pred_idx].detach().cpu().item())
                row = {
                    "id": (int(ids[i]) if ids is not None else None),
                    "true_product_idx": true_idx,
                    "pred_product_idx": pred_idx,
                    "pred_confidence": conf,
                    "true_product_name": self.idx2product.get(true_idx, str(true_idx)),
                    "pred_product_name": self.idx2product.get(pred_idx, str(pred_idx)),
                    "is_correct": true_idx == pred_idx

                }
                rows.append(row)

        return rows, y_true, y_pred

    # -----------------------------
    # Merge with labeled CSV
    # -----------------------------
    def _merge_with_metadata(self, df_preds: pd.DataFrame) -> pd.DataFrame:
        """
        Merge predictions with labeled CSV (original metadata) by ID.
        - Tries 'id_col_meta' (default 'id'); if not found, falls back to 'chat_id'.
        - Ensures proper typing for join.
        - If issue labels are numeric in metadata, adds textual names via idx2issue.
        """
        df_meta = pd.read_csv(self.labeled_csv_path)

        meta_key = self.id_col_meta

        # unify join key types
        df_preds["id"] = df_preds["id"].astype(str)
        df_meta[meta_key] = df_meta[meta_key].astype(str)

        merged = df_preds.merge(df_meta, left_on="id", right_on=meta_key, how="left", suffixes=("", "_meta"))

        return merged

    # -----------------------------
    # Confusion Matrix / Report
    # -----------------------------
    def _save_confusion_matrix(self, y_true, y_pred, out_path="confusion_matrix.png"):
        cm = confusion_matrix(y_true, y_pred)
        labels = [self.idx2product.get(k, str(k)) for k in sorted(self.idx2product)]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap="Blues")
        plt.title("Confusion Matrix (Product)")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print(f"Saved: {out_path}")

    def _save_classification_report(self, y_true, y_pred, out_path):
        print("\nClassification Report:\n")
        target_names = [self.idx2product.get(k, str(k)) for k in sorted(self.idx2product)]
        try:
            # y_true/y_pred may contain only a subset; ignore labels not present
            report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
        except Exception:
            report = classification_report(y_true, y_pred, zero_division=0)
        with open(out_path, "w") as f:
            f.write(report)


    # -----------------------------
    # Orchestrate
    # -----------------------------
    def run(self):
        print("Loading model...")
        self.load_model()
        print("Loading test dataset...")
        self.load_test_dataset()

        print("Running predictions...")
        rows, y_true, y_pred = self._run_predictions()
        df_preds = pd.DataFrame(rows)


        self._save_classification_report(y_true, y_pred, out_path="reports/classification_report.txt")

        if self.save_confusion_matrix:
            self._save_confusion_matrix(y_true, y_pred, out_path="reports/confusion_matrix.png")

        print("Merging with labeled metadata...")
        merged = self._merge_with_metadata(df_preds)

        merged.to_csv(self.output_csv_path, index=False)
        print(f"Saved merged validation CSV: {self.output_csv_path} (rows: {len(merged)})")


# -----------------------------
# Example CLI-style usage
# -----------------------------
if __name__ == "__main__":
    exporter = ValidationExporter(
        model_ckpt_path="../training/lightning_logs/version_6/checkpoints/best_model.ckpt",
        test_pt_path="../data_preprocessing/processed_data/test_encodings.pt",
        labeled_csv_path="../data_preprocessing/raw_data/synthetic_chat_dataset_labeled.csv",
        output_csv_path="reports/validation_full_test_with_preds.csv",
        id_col_meta="chat_id",
        batch_size=16,
        save_confusion_matrix=True,
    )
    exporter.run()
