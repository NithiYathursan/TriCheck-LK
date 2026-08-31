# 🔎 TriCheck-LK

> **A multilingual NLP decision-support system for cross-lingual semantic consistency and missing-information detection in Sinhala, Tamil and English Sri Lankan government documents.**

TriCheck-LK compares corresponding trilingual PDF documents, aligns semantically related sections, detects potential missing-information regions, and presents review-focused results through an interactive Streamlit interface.

---

## 🚀 Key Features

- 📄 **Trilingual PDF Analysis:** Upload corresponding English, Sinhala and Tamil versions of the same government document for automated cross-language comparison.

- 🔍 **Hybrid Text Extraction:** Uses PyMuPDF for direct PDF text extraction with Tesseract OCR fallback for scanned or insufficiently extractable documents.

- 🌐 **Unicode-Safe Multilingual Processing:** Cleans and normalizes English, Sinhala and Tamil text while preserving language-specific Unicode characters and Sinhala ZWJ/ZWNJ behavior.

- ✂️ **Sentence-Aware Chunking:** Divides long documents into semantic chunks of up to 1,000 characters while preserving document order and numbered sections.

- 🧠 **Multilingual Semantic Embeddings:** Uses `intfloat/multilingual-e5-small` to represent English, Sinhala and Tamil text in a shared semantic vector space.

- 🔗 **Order-Aware Semantic Alignment:** Uses Dynamic Time Warping (DTW) to align multilingual document sections even when translations contain different numbers of sentences or chunks.

- 🧪 **Controlled Missing-Information Simulation:** Creates supervised training and evaluation examples by introducing controlled omissions into complete real government circulars.

- 🤖 **Language-Specific ML Detectors:** Uses separate optimized classifiers for English, Sinhala and Tamil rather than forcing a single model across all three languages.

- 🚩 **Flagged Review Regions:** Highlights potentially inconsistent or missing sections and displays corresponding language regions side-by-side for human review.

- ✅ **Document Validation:** Detects unreadable files, insufficient text, empty documents and PDFs uploaded into the wrong language slot.

- 🔄 **Reset Documents:** Allows users to clear the current document set and immediately begin another trilingual comparison.

- 🖥️ **Interactive Streamlit Interface:** Provides a practical end-user application without requiring users to interact with notebooks or model code.

---

## 📊 Dataset

The project uses **official Sri Lankan Public Administration circulars** as its primary domain.

🌐 **Official Source:**  
https://pubad.gov.lk/web/index.php?lang=en&option=com_circular&view=circulars

| Dataset Stage | Count |
|---|---:|
| Circular records collected | 2,109 |
| Complete trilingual records | 2,069 |
| NLP candidates | 2,010 |
| Circulars entering full extraction | 2,008 |
| Final complete trilingual circulars | 1,552 |
| Final language documents | 4,656 |
| Final multilingual chunks | 34,316 |

---

## 🧪 Why Synthetic Missing Information?

The document text used by TriCheck-LK comes from **real Sri Lankan government circulars**.

However, the original repository does not provide verified labels showing exactly where information is missing between language versions.

To obtain reliable ground truth for supervised learning, controlled sections were deliberately removed from complete trilingual documents.

> **Synthetic refers only to the introduced missing-information scenarios — not to the underlying government text.**

Final multilingual experiment:

- 📑 100 real circulars
- 🌐 3 target languages
- 🧪 900 controlled scenarios
- 📊 12,480 feature rows
- 🚩 2,783 synthetic missing-information instances

---

## ⚙️ NLP Workflow

```text
Official Government Circulars
          │
          ▼
     Web Scraping
          │
          ▼
 Metadata & PDF Validation
          │
          ▼
 Hybrid PDF Extraction
   ┌──────┴──────┐
   ▼             ▼
PyMuPDF      Tesseract OCR
   └──────┬──────┘
          ▼
 Unicode-Safe Cleaning
          │
          ▼
 Sentence-Aware Chunking
          │
          ▼
 Multilingual E5 Embeddings
          │
          ▼
 Dynamic Time Warping
          │
          ▼
 Feature Engineering
          │
          ▼
 Language-Specific Models
          │
          ▼
 Flagged Review Regions
          │
          ▼
   Streamlit Application
```

---

## 🧠 Semantic Model

TriCheck-LK uses:

```text
intfloat/multilingual-e5-small
```

The model converts multilingual text into **384-dimensional semantic embeddings**, allowing text with similar meanings in English, Sinhala and Tamil to obtain comparable numerical representations.

This enables cross-language comparison without translating every document into English.

---

## 🔗 Cross-Lingual Alignment

Direct chunk-index comparison is not reliable because multilingual translations may have different structures.

TriCheck-LK therefore uses **Dynamic Time Warping (DTW)** to:

- preserve document order;
- align unequal chunk sequences;
- support one-to-many relationships;
- locate semantically corresponding regions.

---

## 🤖 Final Models

| Target Language | Model | Decision Threshold |
|---|---|---:|
| 🇬🇧 English | HistGradientBoosting | 0.40 |
| 🇱🇰 Sinhala | SVM (RBF) | 0.40 |
| 🇱🇰 Tamil | Random Forest | 0.50 |

The final model configuration is stored in:

```text
models/tricheck_multilingual_config.json
```

---

## 📈 Final Performance

| Language | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| English | 0.6780 | 0.3742 | **0.8182** | 0.5136 |
| Sinhala | 0.7838 | 0.5705 | 0.6268 | 0.5973 |
| Tamil | **0.8126** | **0.6101** | 0.6978 | **0.6510** |

### Performance Summary

- 🟢 **Tamil** achieved the strongest overall balance and highest F1-score.
- 🟡 **Sinhala** produced comparatively balanced precision and recall.
- 🔵 **English** achieved high recall but lower precision, meaning it identifies many potential omissions while producing more false-positive review warnings.

For this reason, TriCheck-LK presents predictions as **review recommendations**, not definite missing-information claims.

---

## 🧮 Confusion Matrices

### English — Fresh Holdout

```text
[[538 301]
 [ 40 180]]
```

### Sinhala

```text
[[346  67]
 [ 53  89]]
```

### Tamil

```text
[[354  62]
 [ 42  97]]
```

---

## 🖥️ Application Features

Users upload:

- 🇬🇧 English PDF
- 🇱🇰 Sinhala PDF
- 🇱🇰 Tamil PDF

The application automatically performs:

**PDF Extraction → Validation → Cleaning → Chunking → Embedding → DTW Alignment → Feature Generation → Prediction → Review Display**

The output includes:

- **No issue detected**
- **Review recommended**
- **Flagged Review Regions**
- side-by-side multilingual section comparison
- model scores for flagged regions

---

## 🛡️ Input Validation

TriCheck-LK validates documents before model analysis.

It can detect:

- ❌ unreadable PDFs;
- ❌ insufficient usable text;
- ❌ blank extracted content;
- ❌ documents producing no valid chunks;
- ❌ documents uploaded into the wrong language section.

For example, a Sinhala PDF uploaded into the Tamil uploader can be detected and rejected before semantic analysis.

---

## 🛠️ Technology Stack

- 🐍 **Python**
- 🖥️ **Streamlit**
- 📄 **PyMuPDF**
- 🔤 **Tesseract OCR**
- 🤗 **Sentence-Transformers**
- 🧠 **multilingual-e5-small**
- 🤖 **Scikit-learn 1.8.0**
- 🐼 **Pandas**
- 🔢 **NumPy**
- 💾 **Joblib**
- 📓 **Jupyter Notebook**
- 🌿 **Git & GitHub**

---

## 📁 Project Structure

```text
TriCheck_LK/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── models/
│   ├── tricheck_english_detector.joblib
│   ├── tricheck_sinhala_detector.joblib
│   ├── tricheck_tamil_detector.joblib
│   └── tricheck_multilingual_config.json
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_semantic_model.ipynb
│   ├── 04_missing_information.ipynb
│   └── 05_evaluation.ipynb
│
├── src/
│   ├── collect_data.py
│   ├── validate_metadata.py
│   ├── validate_pdf_links.py
│   ├── extract_documents.py
│   └── validate_extracted_documents.py
│
└── data/
    ├── metadata/
    └── processed/
```

---

## ⚡ Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd TriCheck_LK
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate it

Windows:

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

---

## 🔤 Tesseract OCR Setup

Tesseract OCR must be installed separately.

Required language packs:

```text
eng
sin
tam
```

The current Windows development configuration uses:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

If Tesseract is installed elsewhere, update the path in `app.py`.

---

## ▶️ Run the Application

From the project root:

```bash
streamlit run app.py
```

or:

```bash
python -m streamlit run app.py
```

The TriCheck-LK interface will open in the browser.

---

## ⚠️ Limitations

- Synthetic omissions are used as ground truth instead of human-annotated real-world missing-information labels.
- OCR quality may affect semantic representations.
- Some Tamil PDFs contain problematic direct-text font or character encoding.
- Chunk boundaries are approximate semantic regions rather than exact legal or translation units.
- The English detector produces comparatively more false-positive warnings.
- The system was primarily trained and evaluated using Sri Lankan Public Administration circulars.
- Performance on other document domains has not been comprehensively validated.
- Model scores should not be interpreted as guaranteed probabilities.
- TriCheck-LK cannot certify legal or linguistic equivalence.

---

## 🎯 Intended Use

TriCheck-LK is designed as a:

> **Multilingual document screening and decision-support tool**

It aims to reduce manual comparison effort by identifying document regions that may require additional attention.

It is **not** intended to replace professional translation review or serve as an automatic legal or administrative certification system.

---

## 🔮 Future Work

- Human-annotated multilingual benchmark datasets
- Improved English precision
- Better Tamil PDF encoding-quality detection
- Advanced OCR quality assessment
- Paragraph- and clause-level semantic segmentation
- Probability calibration
- Explainable missing-information detection
- Additional Sri Lankan government document sources
- Public web deployment
- Document-level semantic consistency scoring

---

## 👤 Author

**N. Yathursan**  
BSc Data Science  
Sabaragamuwa University of Sri Lanka

---

## 📌 Disclaimer

TriCheck-LK was developed as an academic NLP project.

All outputs should be interpreted as **automated review recommendations**. Important administrative documents should still undergo appropriate human verification.
