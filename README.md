# 📰 AI-Powered News Article Summarizer using LoRA Fine-Tuning

## 📌 Problem Description

With the rapid growth of online news platforms, users are constantly overwhelmed by large volumes of lengthy news articles published every day. Reading full articles can be time-consuming, especially when users only need the most important information quickly.

This project aims to develop an AI-powered system capable of automatically generating concise and meaningful summaries from long news articles using advanced Natural Language Processing (NLP) techniques.

The system leverages Transformer-based deep learning models along with LoRA (Low-Rank Adaptation) fine-tuning to efficiently summarize lengthy news content while preserving important contextual information.

By converting long articles into short and informative summaries, the project helps users save time, improve information consumption, and quickly understand the key points of news reports.

---

## 🖼 System Demonstration

<table>
<tr>
<td>

<img src="images/input_photo.png" width="400">

<p align="center"><b>Input News Article</b></p>

</td>

<td>

<img src="images/output_photo.png" width="400">

<p align="center"><b>Generated Summary</b></p>

</td>
</tr>
</table>

---


## 📊 Dataset

For training and evaluating the summarization model, this project uses the **CNN/DailyMail** dataset available on 🤗 Hugging Face.

Dataset link: https://huggingface.co/datasets/cnn_dailymail
Original paper: https://arxiv.org/abs/1506.03340

The CNN/DailyMail dataset is one of the most widely used benchmarks for **abstractive text summarization** tasks. It contains news articles collected from CNN and Daily Mail websites, along with human-written summaries that highlight the key points of each article.

The dataset consists of approximately:

* **287,000 training samples**
* **13,000 validation samples**
* **11,000 test samples**

Each sample includes a news article and its corresponding summary.

### Dataset Structure

| id      | article                                                                                                                 | highlights                                                                |
| ------- | ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| train_1 | The economy is showing signs of recovery after months of uncertainty. Analysts believe the market may stabilize soon... | Analysts say the economy may recover soon after months of uncertainty.    |
| train_2 | Scientists have discovered a new species of deep-sea fish living near hydrothermal vents in the Pacific Ocean...        | Researchers discover a new deep-sea fish species near hydrothermal vents. |
| ...     | ...                                                                                                                     | ...                                                                       |

* **article** → The full news article text
* **highlights** → The ground-truth summary written by humans

These article–summary pairs allow the model to learn how to generate concise summaries while preserving the most important information from long news articles.


## ⚙️ Methodology

### Pre-trained Language Model

This project utilizes the powerful transformer-based model **BART**, specifically the **BART Large CNN** checkpoint, which is widely used for abstractive text summarization tasks.

BART combines the strengths of bidirectional encoders (similar to BERT) and autoregressive decoders (similar to GPT). This hybrid architecture allows the model to effectively understand long textual contexts and generate fluent summaries.

The pre-trained model already possesses strong language understanding capabilities. However, to adapt it specifically for summarizing news articles, it needs to be fine-tuned on a task-specific dataset.

---

### Fine-Tuning Techniques

Instead of using traditional full fine-tuning, which requires updating all model parameters and consumes large computational resources, this project applies **Parameter Efficient Fine-Tuning (PEFT)** methods.

These techniques allow the model to adapt to new tasks while modifying only a small subset of parameters.

#### 1️⃣ Parameter Efficient Fine-Tuning (PEFT)

The project utilizes the **PEFT** framework to efficiently adapt the pre-trained model. PEFT methods significantly reduce training time, GPU memory consumption, and storage requirements compared to full fine-tuning.

---

#### 2️⃣ Low-Rank Adaptation (LoRA)

**Low-Rank Adaptation** introduces small trainable low-rank matrices into specific layers of the transformer model.

Instead of updating the entire weight matrices of the model, LoRA learns small parameter updates that are injected into the attention layers. This allows the model to learn task-specific behavior with only a small number of additional parameters.

Key advantages of LoRA include:

* Reduced GPU memory usage
* Faster training time
* Smaller model checkpoints
* Efficient adaptation of large language models

---

#### 3️⃣ Quantized LoRA (QLoRA)

To further optimize memory usage and enable training on limited hardware resources such as **Google Colab GPUs**, this project also explores **QLoRA**.

QLoRA combines model quantization with LoRA adapters by loading the base model in **4-bit precision** while training only the LoRA parameters. This drastically reduces memory consumption while maintaining strong model performance.

---

### Training Strategy

The training process follows a **sequential adaptation approach**:

1. Load the pre-trained **BART-Large-CNN** model.
2. Apply **LoRA adapters** to the attention layers of the transformer.
3. Fine-tune the model on the **CNN/DailyMail summarization dataset**.
4. Merge the trained LoRA adapters with the base model weights.
5. Optionally continue training using **QLoRA** for further adaptation.

This approach allows the model to learn summarization patterns efficiently while operating within limited computational resources.

---

### Deployment

After fine-tuning, the final model is deployed using **Streamlit**, enabling users to interact with the summarization system through a simple web interface. Users can paste long news articles, and the system generates concise summaries in real time.



---

## 🧠 Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- LoRA / PEFT Fine-Tuning
- Streamlit
- BART Transformer Model
- NLP & Deep Learning

---

## 🏗️ Model Architecture

The project uses a Transformer-based Seq2Seq summarization model fine-tuned using LoRA (Low-Rank Adaptation), which significantly reduces training memory requirements while maintaining strong summarization performance.

Base Model:
- `facebook/bart-large-cnn`

Fine-Tuning Technique:
- LoRA / QLoRA

---

## 💻 Run Locally

### Clone Repository

```bash
git clone https://github.com/Mustafiz004/news_article_summarizer_with_lora.git



