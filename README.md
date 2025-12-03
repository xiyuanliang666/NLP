# NLP
This repository contains the experimental implementations and results for AI6127 course assignments, focusing on two core directions: **deep learning models for sentiment classification** and **Seq2Seq-based machine translation models**. It includes detailed experimental processes, result statistics, and in-depth architectural and parameter analysis.

## Assignment 1: Deep Learning Models for Sentiment Classification
### Core Work
1. Tested three optimizers (SGD, Adam, Adagrad) for sentiment classification, conducted hyperparameter tuning for each, and recorded training, validation, and test metrics.
2. Used the Adam optimizer to explore the impact of different epoch numbers (5, 10, 20, 50) on model performance.
3. Implemented six models (3 FFNN variants, CNN, LSTM, Bi-LSTM) with random initialized embeddings, unified dropout (0.5) and adjusted hidden dimensions for fair comparison.

### Key Results
1. After tuning, the optimizer performance ranking remained **Adam > Adagrad > SGD**, with Adam achieving the highest test accuracy (79.43%), Adagrad at 72.85%, and SGD at 66.99%.
2. The model reached optimal performance at 10 epochs; additional epochs did not improve metrics like train/test accuracy and loss.
3. Among models, Bi-LSTM achieved the highest validation accuracy (88.93%) and test accuracy (88.29%), followed by LSTM (88.89% val acc, 88.20% test acc), while FFNN variants had lower test accuracy (85.70%-86.11%).

### Analysis & Findings
1. **Optimizer Characteristics**: Adam excels due to first-order and second-order momentum, adapting to sparse features in sentiment classification; SGD lacks adaptive learning rates, leading to slow convergence and poor performance; Adagrad’s monotonically decreasing learning rate slows late training.
2.  **Epoch Impact**: The model converges quickly with Adam, and excessive epochs are ineffective due to large learning rates and insufficient model complexity.
3.  **Model Adaptability**: FFNNs cannot capture sequential or local dependencies, leading to overfitting with increased layers; CNN outperforms FFNNs by extracting local phrase features via convolutional kernels; Bi-LSTM’s bidirectional advantage is not obvious due to short task sentences, with performance nearly matching unidirectional LSTM.

## Assignment 2: Seq2Seq Architecture for Machine Translation
### Core Work
1. Built an English-French translation model based on the Seq2Seq framework, using the fra-eng parallel corpus (filtered to sentences ≤15 words) and ROUGE scores for evaluation.
2. Conducted five experiments: baseline (GRU-GRU), LSTM replacing GRU (LSTM-LSTM), encoder replaced with Bi-LSTM (Bi-LSTM-GRU), adding Bahdanau Attention, and encoder replaced with Transformer.
3. Unified experimental settings (hidden dimension 256, teacher-forcing ratio 0.5, SGD optimizer) with adjusted learning rate (0.001) for the Transformer experiment.

### Key Results
1. The attention-enhanced model (GRU-AttnDecoder) performed best, with Rouge-1 F1 (0.6334) and Rouge-2 F1 (0.4555) exceeding the baseline.
2. LSTM-LSTM and Bi-LSTM-GRU underperformed the baseline, with Rouge-1 F1 dropping by 0.0129 and 0.0142 respectively.
3. The Transformer-encoder model failed severely, with Rouge-1 F1 only 0.1328, and further deteriorated with more epochs.

### Analysis & Findings
1. **Architectural Complexity vs. Performance**: GRU outperforms LSTM in short-sequence tasks due to fewer parameters and faster convergence; Bi-LSTM’s bidirectional context is lost when compressed to a single vector for the decoder.
2.  **Attention Mechanism Value**: Bahdanau Attention solves the Seq2Seq bottleneck by enabling dynamic focus on encoder sequences, improving both lexical accuracy and phrase coherence, with more significant gains in Rouge-2 (bigram overlap).
3.  **Transformer Mismatch**: The Transformer encoder is incompatible with the GRU decoder; pooling token-level outputs into a single vector causes severe information loss, and it fails to train stably without large datasets and specialized regularization.
4.  **Model Bias**: Baseline GRU-GRU balances precision and recall; LSTM-LSTM and Bi-LSTM-GRU suffer from reduced recall; the attention model enhances both precision and recall; the Transformer model has severe semantic mapping errors and recall collapse.
