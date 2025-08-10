import pandas as pd

# -----------------------------
# Label mappings (string -> int)
# -----------------------------
product_map = {
    "Optimizer": 0,
    "Inverter": 1,
    "Battery": 2,
    "Other": 3
}

issue_map = {
    "Malfunction": 0,
    "RMA/Logistics": 1,
    "Information Request": 2,
    "Other": 3
}

class ManualLabeling:
    """
    Add human-readable labels (product and issue) to the dataset and
    derive numeric label columns using predefined mappings.

    Steps:
        1) Load input CSV into a DataFrame.
        2) Derive product_type and issue_type from conversation_text.
        3) Map product_type/issue_type to numeric columns (product_label, issue_label).
        4) Save the labeled dataset to CSV.
    """

    def __init__(self, input_file: str, output_file: str):
        """
        Initialize the labeling pipeline.

        Args:
            input_file (str): Path to the cleaned dataset CSV file.
            output_file (str): Path to save the labeled CSV file.
        """
        self.input_file = input_file
        self.output_file = output_file
        self.data: pd.DataFrame | None = None

    # -----------------------------
    # I/O
    # -----------------------------
    def load_data(self) -> None:
        """
        Load the input CSV into self.data.
        """
        self.data = pd.read_csv(self.input_file)
        print(f"[ManualLabeling] Loaded dataset with {len(self.data)} rows from: {self.input_file}")

    def save_labeled_data(self) -> None:
        """
        Save self.data (with added labels) to output_file.
        """
        assert self.data is not None, "Data is not loaded."
        self.data.to_csv(self.output_file, index=False)
        print(f"[ManualLabeling] Labeled dataset saved to: {self.output_file}")

    # -----------------------------
    # Heuristics (string labels)
    # -----------------------------
    @staticmethod
    def detect_product(text: str) -> str:
        """
        Infer product type from conversation text by simple keyword rules.

        Args:
            text (str): Raw conversation text.

        Returns:
            str: One of {"Optimizer", "Inverter", "Battery", "Other"}.
        """
        t = str(text).lower()

        if "optimizer" in t:
            return "Optimizer"
        elif "inverter" in t:
            return "Inverter"
        elif "battery" in t:
            return "Battery"
        else:
            return "Other"

    @staticmethod
    def detect_issue(text: str) -> str:
        """
        Infer issue type from conversation text by simple keyword rules.

        Args:
            text (str): Raw conversation text.

        Returns:
            str: One of {"Malfunction", "RMA/Logistics", "Information Request", "Other"}.
                 NOTE: These values do NOT match the keys in `issue_map` above.
        """
        t = str(text).lower()

        if any(w in t for w in ["not working", "stopped", "failure", "error", "issue", "problem"]):
            return "Malfunction"
        elif any(w in t for w in ["replacement", "rma", "return", "shipping", "logistics"]):
            return "RMA/Logistics"
        elif any(w in t for w in ["status", "update", "check", "contact", "support"]):
            return "Information Request"
        else:
            return "Other"

    # -----------------------------
    # Label application (add columns)
    # -----------------------------
    def add_labels(self) -> None:
        """
        Create textual labels and their numeric mappings.

        Adds columns:
            - product_type (str)
            - issue_type (str)
            - product_label (int)  -> from product_map
            - issue_label (int)    -> from issue_map (see NOTE about mismatch)
        """
        assert self.data is not None, "Data is not loaded."

        # Derive textual labels from conversation text
        self.data["product_type"] = self.data["conversation_text"].apply(self.detect_product)
        self.data["issue_type"] = self.data["conversation_text"].apply(self.detect_issue)

        # Map to numeric labels
        self.data["product_label"] = self.data["product_type"].map(product_map)
        self.data["issue_label"] = self.data["issue_type"].map(issue_map)

        print("[ManualLabeling] Added product/issue labels (text + numeric).")

    # -----------------------------
    # Orchestration
    # -----------------------------
    def run(self) -> None:
        """
        Execute the full manual labeling flow:
        load → add_labels → save.
        """
        self.load_data()
        self.add_labels()
        self.save_labeled_data()


if __name__ == "__main__":
    processor = ManualLabeling(
        input_file="raw_data/cleaned_dataset.csv",
        output_file="raw_data/synthetic_chat_dataset_labeled.csv"
    )
    processor.run()
