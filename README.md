# A Transformer-Based Pipeline for TCR Disease Classification

This repository contains the code and supplementary materials for our project on classifying T-cell receptor (TCR) sequences into distinct disease states (COVID, cancer, and healthy) using a transformer-based deep learning model.

## Overview

Our project develops and validates a deep learning pipeline that integrates paired-chain TCR data with quantitative clonotype frequency to capture nuanced immunological signatures. By adapting a state-of-the-art BERT-based model for immunogenomics, we aim to both improve disease classification accuracy and provide deeper insights into T cell-mediated immune responses.

## Aim of Project

The central aim of this project is to create a robust machine learning pipeline capable of accurately classifying TCR sequences into three distinct disease states: COVID, cancer, and healthy. Our hypothesis is that by integrating both paired-chain TCR data and quantitative clonotype frequency—serving as a proxy for clonal expansion—we can capture subtle immunological signatures that traditional unpaired approaches overlook. We adapt a transformer-based model, known for its ability to extract rich, contextual representations from sequential data, to the field of immunogenomics. Our work strives to improve classification performance while also deepening our biological understanding of T cell responses, ultimately paving the way for the development of clinically relevant biomarkers.

## Background Research

T cells are the linchpin of the adaptive immune system, responsible for recognizing and responding to pathogens and aberrant cells through highly specific antigen recognition. T-cell receptors (TCRs) are composed of alpha and beta chains whose diversity is generated via the V(D)J recombination process. The hypervariable Complementarity-Determining Region 3 (CDR3) is especially critical as it directly interacts with antigens presented by Major Histocompatibility Complex (MHC) molecules. Early TCR research was limited to unpaired beta chains due to sequencing constraints; however, advances in single-cell sequencing now enable paired-chain analyses, providing a comprehensive view of T cell specificity.

The Observed T cell Receptor Space (OTS) database aggregates millions of paired TCR sequences, offering a rich resource for analyzing complex immunological phenomena such as V gene pairing, public clonotypes, and antigen-specific responses. Prior studies have shown that paired-chain analysis can reveal previously hidden patterns in TCR diversity. Nonetheless, many earlier models did not incorporate quantitative features like clonotype frequency—a measure that reflects clonal expansion and is particularly informative in immune activation (e.g., in COVID-19). Our study focuses on COVID, cancer, and healthy states to overcome these limitations and offer a refined understanding of the immune landscape.

## Methods

Our methodology comprises several key components: data processing, tokenization, model development, training, and evaluation.

### Dataset Processing

- **Data Acquisition:** We collected the OTS dataset, consisting of 1959 files from single-cell 10x genomics sequencing runs, each containing up to 193 columns of data.
- **Key Column Selection:** Seventeen essential columns capturing TCR features and metadata were identified and extracted.
- **Cleaning:** Non-alphabetic characters were removed from the CDR3 sequences of both the alpha and beta chains to ensure data validity.
- **Filtering:** We retained only productive sequences (where both `productive_alpha` and `productive_beta` equal "T") and computed the clonotype frequency for each unique $\alpha\beta$ pair.
- **Splitting:** The final dataset of ~5.8 million rows was stratified and split into 240,000 training rows and 60,000 validation rows, preserving the original disease distribution.

### Tokenization and Dataset Creation

- **Custom Tokenization:** TCR sequences are converted into a numerical format by mapping each letter from A to Z to integers 1 through 26. Two special tokens are defined: `[CLS]` (value 27) to denote the start of a sequence and `[SEP]` (value 28) to separate the alpha and beta chains.
- **Sequence Formatting:** Each combined TCR sequence (formatted as `alpha_beta`) is padded or truncated to a fixed length of 100 tokens to ensure uniformity for batch processing.
- **Dataset Class:** The custom PyTorch `TCRDataset` class handles the loading of preprocessed CSV files, applies tokenization, and generates input IDs, attention masks, and token type IDs. It also integrates clonotype frequency as an additional numerical feature and maps disease labels (`covid`, `cancer`, `healthy`) to integer values (0, 1, 2).

### Model Development and Training

- **Model Architecture:** We adapted a transformer-based (BERT) architecture, creating our custom model `BertWithFrequency`. The model extracts a high-dimensional embedding from the `[CLS]` token of each tokenized TCR sequence.
- **Frequency Integration:** An additional linear layer transforms the scalar clonotype frequency into a dense vector. This vector is concatenated with the `[CLS]` embedding, forming a composite representation that combines contextual sequence information with quantitative measures of clonal expansion.
- **Classification and Loss:** The composite representation is fed into a classifier that outputs logits for the three disease classes. To counteract class imbalance (with COVID TCRs dominating the dataset), we employ a weighted cross-entropy loss function with weights set to 1.0 for COVID, 1.8067 for cancer, and 3.6848 for healthy.
- **Training Setup:** The model is trained using the Hugging Face Trainer, for up to 50 epochs with early stopping (patience of 5 epochs), a batch size of 16, a learning rate of $5 \times 10^{-5}$, and a weight decay of 0.01. This setup is optimized for GPU use and ensures that the best model checkpoint (based on validation loss) is saved.

### Evaluation and Visualization

- **Quantitative Evaluation:** The model is evaluated on a validation set of 60,000 samples using metrics such as accuracy, precision, recall, and F1 score. The overall validation accuracy achieved is 63.36%.
- **Performance Metrics:** The model exhibits high recall (91%) and F1 score (0.75) for COVID TCRs, low recall (18%) and F1 score (0.27) for cancer TCRs, and moderate performance for healthy TCRs (recall 54%, F1 score 0.59).
- **Visualization:** A confusion matrix heatmap is generated to visualize true versus predicted labels, revealing that COVID TCRs are predominantly well-classified while cancer TCRs are often misclassified. Additionally, t-SNE is used to project the high-dimensional `[CLS]` embeddings into 2D space, demonstrating distinct clustering of COVID TCRs and significant overlap for cancer and healthy TCRs.

## Results

Our final model achieved an overall validation accuracy of 63.36%. Detailed performance metrics are as follows:
- **COVID (Label 0):** Precision 0.63, Recall 0.91, F1 score 0.75, Support 32,879.
- **Cancer (Label 1):** Precision 0.60, Recall 0.18, F1 score 0.27, Support 18,198.
- **Healthy (Label 2):** Precision 0.66, Recall 0.54, F1 score 0.59, Support 8,923.

The confusion matrix indicates a strong bias toward correctly predicting COVID TCRs, while cancer TCRs are frequently misclassified as COVID. The t-SNE visualization further supports these findings, showing that COVID TCRs form a distinct, dense cluster, whereas cancer and healthy TCRs exhibit considerable overlap. These results suggest that our model effectively captures the pronounced clonal expansion in COVID but struggles with the subtler, more heterogeneous patterns present in cancer and healthy states.

## Lessons Learned

Our work has yielded several key insights and highlighted certain limitations, which point the way toward future improvements. Meticulous cleaning and integration of diverse metadata were essential for preserving biologically relevant signals, yet challenges with missing or inconsistent data (such as age and T-cell subtype) remain. This suggests that future efforts should prioritize standardizing these variables and possibly incorporating additional sequencing modalities to enrich the dataset.

From a modeling perspective, while the integration of clonotype frequency into our transformer-based model improved its ability to capture quantitative aspects of T cell clonal expansion, the stark discrepancy in performance between COVID and cancer indicates that our current feature set may not fully capture the subtle immunological differences among these conditions. Future iterations could benefit from including additional features like V, D, J gene usage, sequence length, and cell-surface markers. Exploring alternative model architectures, such as deeper transformers or hybrid models, and experimenting with advanced loss functions like focal loss may further mitigate class imbalance and enhance sensitivity to minority classes.

Our comprehensive evaluation, incorporating quantitative metrics, confusion matrices, and t-SNE visualizations, provided a detailed understanding of model performance. The observed overlap in cancer and healthy TCR embeddings underscores the inherent biological variability and the challenge of distinguishing these states using current feature extraction methods. Moreover, our study emphasizes the necessity for more rigorous validation through techniques such as stratified k-fold cross-validation and external dataset testing. Future work should also consider interpretability methods like SHAP and attention weight visualization to gain deeper insights into the features driving model predictions.

## Conclusion

In conclusion, our study demonstrates that a deep learning pipeline integrating paired-chain TCR data with clonotype frequency can classify TCR sequences into COVID, cancer, and healthy states with an overall accuracy of 63.36%. The model’s robust performance in identifying COVID TCRs is largely attributable to the pronounced clonal expansion characteristic of the SARS-CoV-2 response. However, the significant challenge of accurately classifying cancer TCRs—evidenced by low recall and F1 scores—highlights the need for further enhancements in both data processing and model architecture. 

Looking ahead, future work should focus on enhanced data processing by incorporating additional immunogenomic features and metadata, as well as on expanding validation using external datasets and rigorous cross-validation techniques. Refinements to the model architecture, such as exploring deeper transformers or hybrid approaches and experimenting with advanced loss functions, will be essential to improve performance, especially for cancer and healthy TCRs. Additionally, aligning model predictions with biological experiments and established literature will be critical for translating our computational findings into clinically actionable insights. Through these improvements, we aim to further unravel the complex language of T cell receptors and advance the development of precise immune biomarkers.

## References
\nocite{*}
\printbibliography

## Appendix

### Jupyter Notebook
The complete Jupyter Notebook containing the Python code for data processing, model training, evaluation, and visualization is included in the supplementary materials. This notebook provides detailed code for:
- Loading and cleaning the OTS dataset.
- Conducting exploratory data analysis (EDA) on both metadata and processed TCR sequence data.
- Tokenization and creation of the custom `TCRDataset` class.
- Development and training of the custom BERT-based model (`BertWithFrequency`) with weighted loss.
- Evaluation of model performance, including generation of confusion matrix heatmaps and t-SNE visualizations.

Please refer to the attached notebook file for a comprehensive view of the code and key model parameters.

\end{document}
