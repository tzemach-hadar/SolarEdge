import pandas as pd
import re
import unicodedata
import spacy
from nltk.corpus import stopwords
import nltk

# Ensure NLTK stopwords are downloaded
nltk.download('stopwords')


class DataPreprocessor:
    """
    A class for cleaning and preprocessing chat conversation data.

    Steps:
    1. Load raw CSV.
    2. Remove duplicates and invalid timestamps.
    3. Normalize and clean text.
    4. Create derived features.
    5. Save cleaned dataset.
    """

    def __init__(self, input_path: str, output_file: str):
        """
        Initialize the preprocessor with file paths and resources.

        Args:
            input_path (str): Path to input CSV file.
            output_file (str): Path to save cleaned CSV.
        """
        self.input_path = input_path
        self.output_file = output_file
        self.data = None
        self.stop_words = set(stopwords.words('english'))
        self.contractions = self._get_contractions()
        self.nlp = spacy.load("en_core_web_sm")

    # -------------------- Load --------------------
    def load_data(self):
        """Load dataset from CSV into a DataFrame."""
        self.data = pd.read_csv(self.input_path)

    # -------------------- Data checks --------------------
    def check_missing_values(self):
        """Return the count of missing values per column."""
        return self.data.isnull().sum()

    def check_duplicates(self):
        """Return the count of duplicate rows."""
        return self.data.duplicated().sum()

    def remove_duplicates(self):
        """Remove duplicate rows from the dataset."""
        self.data = self.data.drop_duplicates()

    def check_invalid_timestamps(self):
        """
        Identify rows with invalid timestamp formats.

        Expected format: MM/DD/YYYY (optionally with time).
        """
        return self.data[~self.data['timestamp'].astype(str).str.match(
            r'\d{1,2}/\d{1,2}/\d{4}', na=False
        )]

    def remove_invalid_timestamps(self):
        """Remove rows with invalid timestamps and convert to datetime."""
        invalid_rows = self.check_invalid_timestamps()
        self.data = self.data.drop(invalid_rows.index)
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])

    # -------------------- Text normalization --------------------
    def _get_contractions(self):
        """Return a dictionary mapping contractions to their expanded forms."""
        return {
            "i'm": "i am", "you're": "you are", "he's": "he is",
            "she's": "she is", "it's": "it is", "we're": "we are",
            "they're": "they are", "don't": "do not", "doesn't": "does not",
            "didn't": "did not", "can't": "cannot", "couldn't": "could not",
            "won't": "will not", "wouldn't": "would not", "shouldn't": "should not",
            "i've": "i have", "you've": "you have", "we've": "we have",
            "they've": "they have", "i'll": "i will", "you'll": "you will",
            "he'll": "he will", "she'll": "she will", "we'll": "we will",
            "they'll": "they will", "isn't": "is not", "aren't": "are not",
            "wasn't": "was not", "weren't": "were not", "haven't": "have not",
            "hasn't": "has not", "hadn't": "had not",
            "let's": "let us", "y'all": "you all"
        }

    def expand_contractions(self, text: str) -> str:
        """Expand all contractions in a text string."""
        pattern = re.compile(
            '({})'.format('|'.join(map(re.escape, self.contractions.keys()))),
            flags=re.IGNORECASE
        )

        def replace(match):
            return self.contractions[match.group(0).lower()]

        return pattern.sub(replace, text)

    def normalize_text(self, text: str) -> str:
        """
        Normalize conversation text.

        Steps:
        1. Remove HTML tags.
        2. Lowercase text.
        3. Expand contractions.
        4. Remove punctuation and accents.
        5. Lemmatize and remove stopwords.
        """
        text = str(text)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = text.lower()
        text = self.expand_contractions(text)
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

        doc = self.nlp(text)
        tokens = [
            t.lemma_ for t in doc
            if t.text not in self.stop_words and t.lemma_.strip()
        ]
        return ' '.join(tokens)

    def clean_conversation_column(self):
        """Create 'clean_conversation' column from 'conversation_text'."""
        self.data['clean_conversation'] = self.data['conversation_text'].astype(str).apply(
            self.normalize_text
        )

    # -------------------- Feature engineering --------------------
    def create_message_count_feature(self):
        """Count the number of messages in each conversation."""
        self.data['message_count'] = self.data['clean_conversation'].astype(str).apply(
            lambda x: len(re.findall(r'\b(agent|customer)\b\s+\d+\b', x))
        )

    def create_text_length_feature(self):
        """Count the number of words in each conversation."""
        self.data['text_length'] = self.data['clean_conversation'].apply(
            lambda x: len(str(x).split())
        )

    # -------------------- Save --------------------
    def save_cleaned_data(self, output_path: str):
        """Save cleaned dataset to CSV."""
        self.data.to_csv(output_path, index=False)

    # -------------------- Pipeline runner --------------------
    def run(self):
        """Execute the full preprocessing pipeline."""
        self.load_data()

        # Remove duplicates
        num_duplicates = self.check_duplicates()
        if num_duplicates > 0:
            print(f"Found {num_duplicates} duplicate rows. Removing...")
            self.remove_duplicates()

        # Remove invalid timestamps
        invalid_count = len(self.check_invalid_timestamps())
        if invalid_count > 0:
            print(f"Found {invalid_count} invalid timestamps. Removing...")
            self.remove_invalid_timestamps()

        # Clean text and create features
        print("Cleaning conversation text...")
        self.clean_conversation_column()

        print("Creating message count feature...")
        self.create_message_count_feature()

        print("Creating text length feature...")
        self.create_text_length_feature()

        # Save result
        self.save_cleaned_data(self.output_file)
        print(f"Preprocessing complete. Saved to {self.output_file}")


def main(input_file: str, output_file: str):
    """Run preprocessing from start to finish."""
    processor = DataPreprocessor(input_file, output_file)
    processor.run()


if __name__ == "__main__":
    main(
        "raw_data/synthetic_chat_dataset_for_home_assigment.csv",
        "raw_data/cleaned_dataset.csv"
    )