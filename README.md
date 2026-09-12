# 🔎 TriCheck-LK

> **A multilingual NLP decision-support system for cross-lingual semantic consistency and missing-information detection in Sinhala, Tamil and English Sri Lankan government documents.**

TriCheck-LK compares corresponding trilingual PDF documents, validates that the uploads belong to the correct languages and appear to represent the same source document, aligns semantically related regions, detects potential missing-information areas, and presents review-focused results through an interactive Streamlit interface.

> **Current application behavior:** Model-only review alerts use a conservative minimum threshold of `0.60`, while a separate structural rule can flag confidently missing numbered sections for manual review.

---

## 🚀 Key Features

- 📄 **Trilingual PDF Analysis:** Upload corresponding English, Sinhala and Tamil versions of the same government document for automated cross-language comparison.

- 🌐 **Language-Slot Validation:** Detects whether each uploaded file is predominantly English, Sinhala or Tamil and rejects files placed in the wrong language uploader.

- 🔗 **Same-Document Verification:** Performs pairwise checks across English-Sinhala, English-Tamil and Sinhala-Tamil before semantic analysis. Document references and filename evidence are used when available together with multilingual semantic similarity and coverage metrics.

- 🔍 **Quality-Aware Hybrid Text Extraction:** Uses PyMuPDF for direct PDF text extraction and automatically falls back to Tesseract OCR when direct text is empty, insufficient or appears damaged.

- 🔤 **Complex-Script Quality Checks:** Detects replacement characters, null characters, dotted-circle artifacts and isolated Unicode combining marks that may indicate broken Sinhala or Tamil PDF text layers.

- 🌐 **Unicode-Safe Multilingual Processing:** Cleans and normalizes English, Sinhala and Tamil text while preserving language-specific Unicode content and Sinhala ZWJ/ZWNJ behavior.

- ✂️ **Sentence-Aware Chunking:** Divides long documents into semantic chunks of up to 1,000 characters while preserving document order and numbered-section cues.

- 🧠 **Multilingual Semantic Embeddings:** Uses `intfloat/multilingual-e5-small` to represent English, Sinhala and Tamil text in a shared semantic vector space.

- 🔗 **Order-Aware Semantic Alignment:** Uses Dynamic Time Warping (DTW) to align multilingual document regions even when translations contain different numbers of sentences or chunks.

- 🧪 **Controlled Missing-Information Simulation:** Creates supervised training and evaluation examples by introducing controlled omissions into complete real government circulars.

- 🤖 **Language-Specific ML Detectors:** Uses separate optimized classifiers for English, Sinhala and Tamil rather than forcing a single model across all three languages.

- 🛡️ **Conservative Review Threshold:** The application uses a minimum review threshold of `0.60` for model-only alerts, reducing low-confidence review recommendations while retaining each model's raw score for inspection.

- 🔢 **Structural Numbered-Section Check:** If a numbered section exists in both other language versions but is missing between adjacent sections in the target language, the system independently recommends manual review.

- 🚩 **Review-Focused Output:** Shows review scores and side-by-side multilingual regions. When a numbered section is structurally absent, the UI explicitly reports that it was not detected instead of displaying an unrelated nearest-matching chunk.

- 🔄 **Reset Documents:** Allows users to clear the current document set and immediately begin another trilingual comparison.

- 🖥️ **Interactive Streamlit Interface:** Provides a practical end-user application without requiring users to interact with notebooks or model code.

---

## 📊 Dataset

The project uses **official Sri Lankan Public Administration circulars** as its primary domain.

🌐 **Official Source:**  
https://pubad.gov.lk/web/index.php?lang=en&option=com_circular&view=circulars

| Dataset Stage | Count |
|---|---:|
| Circular records collected | 2109 |
| Complete trilingual records | 2069 |
| NLP candidates | 2010 |
| Circulars entering full extraction | 2008 |
| Final complete trilingual circulars | 1552 |
| Final language documents | 4656 |
| Final multilingual chunks | 34316 |

---

## 🧪 Why Synthetic Missing Information?

The document text used by TriCheck-LK comes from **real Sri Lankan government circulars**.

However, the original dataset does not provide verified labels showing exactly where information is missing between language versions.

To obtain reliable ground truth for supervised learning, controlled sections were deliberately removed from complete trilingual documents.

> **Synthetic refers only to the introduced missing-information scenarios — not to the underlying government text.**

### Final Multilingual Experiment

- 📑 **100** real circulars
- 🌐 **3** target languages
- 🧪 **900** controlled scenarios
- 📊 **12,480** feature rows
- 🚩 **2,783** synthetic missing-information instances

---

## ⚙️ NLP Workflow

```text
English / Sinhala / Tamil PDFs
          │
          ▼
   Language Validation
          │
          ▼
 Same-Document Verification
          │
          ▼
 Quality-Aware PDF Extraction
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
          ├───────────────┐
          ▼               ▼
 Conservative ML     Numbered-Section
 Review Decision     Structural Check
          └───────┬───────┘
                  ▼
        Manual Review Regions
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

The trained model configuration is stored in:

```text
models/tricheck_multilingual_config.json
```

The final Streamlit application applies a conservative minimum review threshold of `0.60` to model-only alerts.

| Target Language | Model | Stored Model Threshold | App Review Threshold |
|---|---|---:|---:|
| 🇬🇧 English | HistGradientBoosting | 0.40 | **0.60** |
| 🇱🇰 Sinhala | SVM (RBF) | 0.40 | **0.60** |
| 🇱🇰 Tamil | Random Forest | 0.50 | **0.60** |

The original model score is retained for review display and analysis.

Separately, a high-confidence structural numbered-section rule can recommend manual review even when the model-only threshold is not reached.

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

- 🇬🇧 **English PDF**
- 🇱🇰 **Sinhala PDF**
- 🇱🇰 **Tamil PDF**

Before running missing-information analysis, the application checks both **language placement** and **document correspondence**.

Pairwise document verification uses circular/reference information where available, filename evidence where useful, and multilingual semantic metrics.

### Application Workflow

```text
PDF Upload
    ↓
Language Validation
    ↓
Same-Document Verification
    ↓
Quality-Aware Extraction
    ↓
Text Cleaning
    ↓
Chunking
    ↓
Multilingual Embeddings
    ↓
DTW Alignment
    ↓
Feature Generation
    ↓
ML Prediction
    ↓
Structural Section Check
    ↓
Review Display
```

### Output Includes

- **No issue detected**
- **Review recommended**
- **Flagged Review Regions**
- **Review score** for each flagged region
- side-by-side multilingual comparison for model-based review regions
- explicit **numbered section not detected** messages when structural checks identify an absent section
- document preparation summary showing:
  - detected language
  - extraction method
  - character count
  - chunk count

---

## 🛡️ Input Validation

TriCheck-LK validates documents before model analysis.

It can detect or stop on:

- ❌ unreadable PDFs
- ❌ insufficient usable text
- ❌ blank extracted content
- ❌ documents producing no valid chunks
- ❌ documents uploaded into the wrong language section
- ❌ uploaded files that do not appear to be corresponding English, Sinhala and Tamil versions of the same document

Same-document verification is performed pairwise across the three language versions.

The system checks:

```text
English ↔ Sinhala
English ↔ Tamil
Sinhala ↔ Tamil
```

If two versions agree strongly and one does not, the application can identify the likely mismatched language.

If the evidence is not strong enough to isolate a single mismatched document, TriCheck-LK reports that **two or more documents may be unrelated** instead of guessing.

---

## 🔍 PDF Text Extraction

TriCheck-LK uses a quality-aware extraction strategy.

### Direct Extraction

PyMuPDF is used first to extract text directly from the PDF.

### OCR Fallback

For Sinhala and Tamil documents, OCR may be used when the direct PDF text appears damaged or unusable.

The application checks for issues such as:

- replacement characters;
- null characters;
- dotted-circle artifacts;
- isolated Unicode combining marks;
- insufficient native-script characters;
- poor extraction quality.

When OCR is required, PDF pages are rendered as images and processed using **Tesseract OCR** through **PyTesseract**.

This approach helps support both:

- text-based PDFs;
- scanned or poorly encoded PDFs.

---

## 🔢 Structural Missing-Section Detection

Model predictions are not the only source of review recommendations.

TriCheck-LK also performs a structural numbered-section check.

For example:

```text
English:
1 2 3 4 5 6 7

Sinhala:
1 2 3 4 5 6 7

Tamil:
1 2 3   5 6 7
```

Because **Section 4** appears in both English and Sinhala but is missing from Tamil while Sections 3 and 5 are present, TriCheck-LK identifies Section 4 as a strong structural omission.

The corresponding target language is then recommended for manual review.

This rule is deliberately conservative and focuses mainly on missing **middle sections**.

---

## 🎯 Review Philosophy

TriCheck-LK does not automatically declare that a translation is correct or incorrect.

Instead, the system identifies **regions that may require additional human attention**.

The final application combines:

- machine-learning predictions;
- multilingual semantic alignment;
- document structure;
- section-number evidence;
- extraction-quality checks.

This makes TriCheck-LK a **decision-support system** rather than an automatic certification tool.

---

## 🛠️ Technology Stack

### Final Streamlit Application

- 🐍 **Python**
- 🖥️ **Streamlit**
- 🔢 **NumPy**
- 🐼 **Pandas**
- 💾 **Joblib**
- 🤖 **Scikit-learn 1.8.0**
- 🤗 **Sentence-Transformers**
- 🧠 **`intfloat/multilingual-e5-small`**
- 📄 **PyMuPDF**
- 🔤 **PyTesseract**
- 🔍 **Tesseract OCR**
- 🖼️ **Pillow**

### Project / Data Pipeline

- 🌐 **Requests**
- 📓 **Jupyter Notebook**
- 🌿 **Git & GitHub**

> `Requests` is mainly used by the data-collection utilities rather than the final Streamlit inference path.

---

## 📦 Python Dependencies

The main Python dependencies include:

```text
streamlit
numpy
pandas
joblib
scikit-learn==1.8.0
sentence-transformers
PyMuPDF
pytesseract
Pillow
requests
```

Install them using:

```bash
python -m pip install -r requirements.txt
```

---

## 📁 Project Structure

```text
TriCheck-LK/
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

### 1. Clone the Repository

```bash
git clone https://github.com/NithiYathursan/TriCheck-LK.git
cd TriCheck-LK
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

#### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

#### Windows Command Prompt

```cmd
.venv\Scripts\activate
```

### 4. Install Dependencies

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

The application first attempts to locate `tesseract` from the system `PATH`.

On Windows, it also checks the common installation location:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

If Tesseract is unavailable, direct-text PDFs can still be processed, but scanned or damaged Sinhala and Tamil PDFs may not be handled reliably.

---

## ▶️ Run the Application

From the project root directory:

```bash
streamlit run app.py
```

or:

```bash
python -m streamlit run app.py
```

The application will normally open at:

```text
http://localhost:8501
```

---

## 📋 Using the Application

1. Upload the **English PDF**.
2. Upload the corresponding **Sinhala PDF**.
3. Upload the corresponding **Tamil PDF**.
4. Click **Check Documents**.
5. Wait while TriCheck-LK:
   - extracts the text;
   - validates the languages;
   - verifies document correspondence;
   - creates embeddings;
   - performs DTW alignment;
   - generates model features;
   - performs ML inference;
   - performs structural checks.
6. Review the **Document Preparation Summary**.
7. Review the **Document Check Result**.
8. Inspect any sections listed under **Sections Recommended for Review**.
9. Use **Reset Documents** to begin a new comparison.

---

## ⚠️ Limitations

- Synthetic omissions are used as ground truth instead of human-annotated real-world missing-information labels.

- OCR quality may affect semantic representations and downstream alignment.

- Some Sinhala and Tamil PDFs contain damaged embedded text layers caused by font or character encoding.

- OCR fallback can improve usability but may still introduce recognition errors.

- Chunk boundaries are approximate semantic regions rather than exact legal or translation units.

- The underlying English detector has comparatively lower precision.

- The final application uses a conservative `0.60` minimum review threshold to suppress low-confidence model alerts, but false positives may still occur.

- The structural numbered-section rule is intentionally conservative and is strongest for missing **middle sections** where both adjacent section numbers are present.

- Same-document verification uses available document references, filenames and semantic evidence.

- Unusual filenames or highly similar unrelated documents can still be challenging.

- The system was primarily trained and evaluated using Sri Lankan Public Administration circulars.

- Performance on other document domains has not been comprehensively validated.

- Review scores should not be interpreted as calibrated probabilities or legal confidence values.

- TriCheck-LK cannot certify legal, administrative or linguistic equivalence.

---

## 🎯 Intended Use

TriCheck-LK is designed as a:

> **Multilingual document screening and decision-support tool**

It aims to reduce manual comparison effort by identifying document regions that may require additional attention.

The system is intended to **support human verification**, not replace it.

TriCheck-LK is **not** intended to:

- replace professional translation review;
- make legal decisions;
- certify linguistic equivalence;
- certify administrative equivalence;
- automatically determine whether an official translation is legally correct.

---

## 🔮 Future Work

Possible future improvements include:

- human-annotated multilingual benchmark datasets;
- improved English precision;
- language-specific threshold calibration;
- improved Sinhala and Tamil PDF text-layer quality assessment;
- advanced OCR quality assessment;
- post-OCR correction for Sinhala and Tamil;
- improved document-reference extraction across different circular formats;
- paragraph-level semantic segmentation;
- clause-level semantic segmentation;
- probability calibration;
- clearer review-score interpretation;
- explainable missing-information detection;
- additional Sri Lankan government document sources;
- public web deployment;
- document-level semantic consistency scoring.

---

## 👤 Author

**N. Yathursan**  
BSc Data Science  
Sabaragamuwa University of Sri Lanka

---

## 📌 Disclaimer

TriCheck-LK was developed as an **academic Natural Language Processing project**.

All outputs should be interpreted as **automated review recommendations**.

Important administrative documents should still undergo appropriate human verification.

The system does not provide legal, linguistic or administrative certification.
