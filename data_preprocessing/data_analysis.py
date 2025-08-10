import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os


class DataAnalysis:
    def __init__(self, input_file: str, output_dir: str = "reports"):
        """
        Initialize the data analysis process.

        Args:
            input_file (str): Path to the cleaned dataset CSV file.
            output_dir (str): Directory where all generated plots/reports will be saved.
        """
        self.input_file = input_file
        self.output_dir = output_dir
        self.data = None

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self):
        """
        Load the cleaned dataset from CSV into a DataFrame.
        """
        self.data = pd.read_csv(self.input_file)

        print("Dataset loaded successfully:")
        print(self.data.info())

    def plot_message_count_distribution(self):
        """
        Plot the distribution of the number of messages per conversation.
        """
        plt.figure(figsize=(8, 5))
        sns.histplot(self.data['message_count'], bins=15, kde=False)

        plt.title("Distribution of Message Counts per Chat")
        plt.xlabel("Number of Messages")
        plt.ylabel("Frequency")
        plt.xticks(range(self.data['message_count'].min(), self.data['message_count'].max() + 1))

        fig_path = os.path.join(self.output_dir, "message_count_distribution.png")
        plt.savefig(fig_path)
        plt.close()

        print(f"Message count distribution saved to {fig_path}")

    def plot_text_length_distribution(self):
        """
        Plot the distribution of conversation lengths (in words).
        """
        plt.figure(figsize=(8, 5))
        sns.histplot(self.data['text_length'], bins=20)

        plt.title("Distribution of Text Length (words)")
        plt.xlabel("Word Count")
        plt.ylabel("Frequency")

        fig_path = os.path.join(self.output_dir, "text_length_distribution.png")
        plt.savefig(fig_path)
        plt.close()

        print(f"Text length distribution saved to {fig_path}")

    @staticmethod
    def detect_product(text: str) -> str:
        """
        Detect product type from text based on keyword presence.

        Args:
            text (str): Conversation text.

        Returns:
            str: Detected product category.
        """
        text = str(text).lower()

        if "optimizer" in text:
            return "Optimizer"
        elif "inverter" in text:
            return "Inverter"
        elif "battery" in text:
            return "Battery"
        else:
            return "Other"

    def plot_product_distribution(self):
        """
        Plot a pie chart showing the distribution of detected products.
        """
        self.data['product'] = self.data['clean_conversation'].apply(self.detect_product)
        product_counts = self.data['product'].value_counts()

        plt.figure(figsize=(8, 8))
        plt.pie(product_counts, labels=product_counts.index, autopct='%1.1f%%',
                startangle=140, colors=sns.color_palette('pastel'))

        plt.title("Chats per Product Category")
        plt.ylabel('')

        fig_path = os.path.join(self.output_dir, "product_distribution_pie.png")
        plt.savefig(fig_path)
        plt.close()

        print(f"Product distribution pie chart saved to {fig_path}")

    def generate_wordcloud(self):
        """
        Generate and save a word cloud from all conversation texts.
        """
        all_text = " ".join(self.data['clean_conversation'].astype(str))

        # Remove common words like "customer" and "agent"
        for word in ["customer", "agent"]:
            all_text = all_text.replace(word, "")

        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title("Most Frequent Words in Chats")

        fig_path = os.path.join(self.output_dir, "wordcloud.png")
        plt.savefig(fig_path)
        plt.close()

        print(f"Word cloud saved to {fig_path}")

    def plot_activity_by_time_and_product(self, time_col: str):
        """
        Plot stacked bar chart showing chat activity by time and product.

        Args:
            time_col (str): Column to group by ("hour" or "date").
        """
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'], errors='coerce')
        self.data['hour'] = self.data['timestamp'].dt.hour
        self.data['date'] = self.data['timestamp'].dt.date

        title = f"Message Volume by {time_col.capitalize()} and Product Type"
        filename = f"activity_by_{time_col}_stacked.png"

        counts = self.data.groupby([time_col, 'product']).size().reset_index(name='count')
        pivot = counts.pivot(index=time_col, columns='product', values='count').fillna(0)
        pivot = pivot.sort_index()

        pivot.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='tab10')

        plt.title(title)
        plt.xlabel(time_col.capitalize())
        plt.ylabel("Number of Messages")
        plt.xticks(rotation=45 if time_col == 'date' else 0)
        plt.legend(title="Product Type")
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path)
        plt.close()

        print(f"Saved stacked chart to {save_path}")

    def plot_top_words_by_product(self, num_words: int = 10):
        """
        Plot bar charts of the most frequent words for each product category.

        Args:
            num_words (int): Number of top words to display.
        """
        if 'product' not in self.data.columns:
            self.data['product'] = self.data['clean_conversation'].apply(self.detect_product)

        products = self.data['product'].unique()
        rows = (len(products) + 1) // 2
        cols = 2 if len(products) > 1 else 1

        fig, axes = plt.subplots(rows, cols, figsize=(15, 6 * rows), squeeze=False)
        axes = axes.flatten()

        for i, product in enumerate(products):
            product_data = self.data[self.data['product'] == product]
            all_text = " ".join(product_data['clean_conversation'].astype(str))
            words = [w for w in all_text.split() if w not in ['customer', 'agent'] and not w.isdigit()]
            word_freq = pd.Series(words).value_counts().head(num_words)

            sns.barplot(x=word_freq.values, y=word_freq.index, hue=word_freq.index, ax=axes[i], palette='viridis', legend=False)
            axes[i].set_title(f'Top {num_words} Words for {product}')
            axes[i].set_xlabel("Frequency")
            axes[i].set_ylabel("Word")

        plt.tight_layout()

        fig_path = os.path.join(self.output_dir, "top_words_by_product.png")
        plt.savefig(fig_path)
        plt.close()

        print(f"Top words per product plot saved to {fig_path}")

    def run_da(self):
        """
        Execute the full data analysis workflow.
        """
        self.load_data()
        self.plot_message_count_distribution()
        self.plot_text_length_distribution()
        self.plot_product_distribution()
        self.plot_activity_by_time_and_product("hour")
        self.plot_activity_by_time_and_product("date")
        self.generate_wordcloud()
        self.plot_top_words_by_product()


def main():
    """
    Main entry point to run the analysis pipeline.
    """
    da = DataAnalysis(input_file="raw_data/cleaned_dataset.csv")
    da.run_da()
    print("Data analysis pipeline completed successfully.")


if __name__ == "__main__":
    main()
