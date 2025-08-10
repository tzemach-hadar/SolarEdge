# Pipeline Overview

### **1\. Data Cleaning**

* **Input:**  
   Raw labeled chat dataset (CSV) containing conversation text, chat\_id, timestemp.  
* **Logic:**  
  * Remove duplicate records.

  * Validate field values and ensure label integrity.

  * Canonicalize text: replace abbreviations with full words.

  * Normalize text:

    * Convert to lowercase.

    * Remove `<br>` tags.

    * Strip special characters, punctuation, or diacritics.

  * Tokenize text using **spaCy** for further processing.

  * Add **interaction\_count** column (number of messages in conversation).

  * Add **conversation\_length** column (character length of conversation).

* **Output:**  
   Clean, structured dataset saved to CSV for labeling and further analysis.


  ---

  ### **2\. Data Analysis**

* ### **Input:**    Cleaned chat dataset (CSV) containing conversation text, metadata, and clean text.

* ### **Logic:**

  ### Perform exploratory data analysis (EDA) to understand dataset characteristics.

  ### Generate multiple visualizations:

  * ### Message count distribution (`plot_message_count_distribution`).

  * ### Conversation length distribution (`plot_text_length_distribution`).

  * ### Product label distribution (`plot_product_distribution`).

  * ### Activity by hour (`plot_activity_by_time_and_product("hour")`).

  * ### Activity by date (`plot_activity_by_time_and_product("date")`).

  * ### Word cloud of most frequent terms (`generate_wordcloud`).

  * ### Top words per product (`plot_top_words_by_product`). 

* ### **Output:**    A set of visual analytics plots providing insights into dataset structure, product distribution, temporal activity patterns, and common vocabulary.


  ---

  ### **3\. Data Labeling**

* **Input:**  
   Cleaned chat dataset (CSV) containing conversation text and metadata.  
* **Logic:**  
  * Apply **naive labeling** based on predefined keyword/product associations.

  * Assign both **product label** and **issue type label** to each conversation.

  * Use `product_map` and `issue_map` to map text labels to integer IDs.

* **Output:**  
   Labeled dataset containing conversation text, metadata, and both numeric \+ textual product/issue labels, saved as CSV for training.


  ---

     **4\. Tokenization**

* **Input:** Labeled dataset (CSV).  
* **Logic:**

  * **Data Split** – Split the dataset into training and validation sets.  
  * **Tokenization** – Use Hugging Face’s `AutoTokenizer` to convert text into `input_ids` and `attention_mask`.

* **Output:** `.pt` files (PyTorch tensors) ready for loading during training and evaluation.


---

### **5\. Dataset Module**

* **Input:** Tokenized `.pt` dataset files (train/test).

* **Logic:**

  * Load tokenized tensors and organize them into `ChatDataset` objects.

  * Provide PyTorch `DataLoader` interfaces for batching during training and evaluation.

* **Output:** Data loaders for train and validation splits.

  ---

  ### **6\. Model Architecture**

* **Input:** Tokenized batches from the data module.

* **Logic:**

  * Load a pretrained DistilBERT model as the encoder.

  * Add a classification head (fully connected layer) for product classification.

  * Forward pass outputs raw logits for each class.

* **Output:** Logits tensor of shape `[batch_size, num_labels]`.

  ---

  ### **7\. Training Module**

* **Input:** Model, data loaders, optimizer settings.

* **Logic:**

  * Training loop: compute cross-entropy loss and accuracy per step.

  * Validation loop: evaluate model performance at the end of each epoch.

  * Save the best checkpoint based on validation accuracy.

  * Track training/validation loss and accuracy for plotting.

* **Output:** Trained model checkpoint \+ training metrics.

  ---

  ### **8\. Model Validation & Testing**

* **Input:** Best model checkpoint, test dataset.

* **Logic:**

  * Load the trained model and run inference on the test set.

  * Compare predictions to ground truth labels.

  * Generate `classification_report` and confusion matrix.

  * Merge predictions with original test dataset for debugging.

* **Output:**

  * CSV file containing full test records \+ predicted labels \+ correctness flag.

  * Classification report and confusion matrix image.

  ---

# Findings

The evaluation results on the test set are as follows:

| Class | Precision | Recall | F1-Score | Support |
| ----- | ----- | ----- | ----- | ----- |
| **Optimizer** | **1.00** | **0.92** | **0.96** | **12** |
| **Inverter** | **0.90** | **1.00** | **0.95** | **9** |
| **Battery** | **1.00** | **1.00** | **1.00** | **14** |
| **Other** | **1.00** | **1.00** | **1.00** | **5** |
| **Overall Accuracy** | **0.97** | **–** | **–** | **40** |

**Interpretation and Insights:**

* **Overall Performance –** The model achieved an accuracy of 97% with very high precision and recall across all classes, indicating strong generalization on the test data.

* **Class-Level Observations:**

  * Optimizer: Precision is perfect (1.00), meaning all predictions labeled as “Optimizer” were correct. However, recall is slightly lower (0.92), indicating that some true “Optimizer” cases were misclassified into another category (in this case, “Inverter”).

  * **Inverter:** Recall is perfect (1.00), meaning the model successfully identified all actual “Inverter” cases. Precision is slightly lower (0.90), meaning some predictions labeled as “Inverter” were actually another product type (mainly “Optimizer”), leading to false positives.

  * **Battery:** Perfect scores across all metrics, showing the model can fully distinguish this class without confusion.

  * **Other:** Also perfect scores, suggesting the class boundaries are well defined.

* **Error Pattern:**

  * After manually checking the error on the test, it seems that in the case of a failed tag, the tagging was incorrect, the tagging was done automatically and naively, and therefore it was tagged as an optimizer and not as an inverter (because both are mentioned in the conversation).  
  * In fact, the tagging should be changed, and this way we will reach 100% accuracy on the test  
* **Data Balance:**

  * All classes achieved high scores despite the smaller support counts for “Other” (5) and “Inverter” (9). This suggests the model is learning generalizable patterns but may still be sensitive to borderline cases in underrepresented categories.

* **Conclusion:**

  * The model is production-ready for accurate classification of product types in support chats, with only minor improvements required in more precise labeling of categories.

# Recommendations for Improvement

1. **Multi-Task Learning –** Extend the model to predict both product type and issue type for richer outputs. This approach can be especially valuable if issue type labels are available, enabling the system to route cases directly to the appropriate support or operational team, thereby reducing resolution time and improving service efficiency.

2. **Improve Label Accuracy** – Use semi-automated methods for tag refinement, such as weak supervision frameworks (e.g., Snorkel) or active learning with human validation.

3. **Increase Data Volume** – Collect more real-world data and apply synthetic data augmentation (e.g., paraphrasing with LLMs, back-translation, simulated chat generation).

4. **Balanced Test Set** – Ensure equal representation of products in the test set for fair evaluation.

5. **Deeper Data Insights** – Produce statistics per product/issue to detect trends and potential production or logistics bottlenecks.

6. **Training Optimization** – Use GPU acceleration and store preprocessed data on GPU to reduce training time.

7. **Hyperparameter Tuning** – Experiment with different learning rates, batch sizes, and optimizer configurations to improve convergence and final accuracy.

8. **Transfer Learning with Frozen Base Model** – Start with a pretrained model, freeze the base layers, and train only the classification head to speed up convergence and reduce overfitting when data is limited.

   

**Business Value**

* **Reduced Customer Wait and Resolution Time:** Automatically route support requests to the right team, enabling faster and more efficient handling.

* **Automated Monitoring of Common Issues:** Identify and track frequent faults or inquiries to enable proactive prevention or improved service.


