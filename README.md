# 🏨 Hotel Reviews Data Mining Project - Đề tài 11

## 📋 Mục lục

1. [Giới thiệu](#giới-thiệu)
2. [Dataset](#dataset)
3. [Cấu trúc Project](#cấu-trúc-project)
4. [Hướng dẫn cài đặt](#hướng-dẫn-cài-đặt)
5. [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
6. [Pipeline](#pipeline)
7. [Kết quả](#kết-quả)
8. [License](#license)

---

## 🌟 Giới thiệu

Đề tài 11 - **Phân tích đánh giá khách sạn & chủ đề dịch vụ**

Dự án này thực hiện phân tích toàn diện dữ liệu đánh giá khách sạn từ Kaggle Hotel Reviews Dataset, bao gồm:

- **EDA (Exploratory Data Analysis)**: Phân tích dữ liệu khám phá
- **Preprocessing**: Tiền xử lý văn bản (làm sạch, rút gọn, mã hóa)
- **Feature Engineering**: Trích xuất đặc trưng TF-IDF, đặc trưng thống kê
- **Association Rules Mining**: Khai phá luật kết hợp giữa các khía cạnh dịch vụ
- **Clustering**: Phân cụm chủ đề đánh giá (K-Means, HDBSCAN)
- **Sentiment Classification**: Phân lớp cảm xúc (positive/neutral/negative)
- **Semi-supervised Learning**: Label Spreading và Self-Training
- **Regression**: Dự đoán rating từ nội dung review
- **Streamlit Demo App**: Ứng dụng web trực quan

---

## 📊 Dataset

### Nguồn dữ liệu
- **Kaggle**: [Hotel Reviews Dataset](https://www.kaggle.com/datasets)
- **Mô tả**: Tập dữ liệu chứa các đánh giá khách sạn với nội dung văn bản và điểm rating

### Data Dictionary

| Column | Mô tả | Kiểu dữ liệu |
|--------|-------|---------------|
| `Review` | Nội dung đánh giá khách sạn | Text |
| `Rating` | Điểm đánh giá (1-5) | Integer |
| `Reviewer` | Tên người đánh giá | String |
| `Date` | Ngày đánh giá | DateTime |
| `Hotel` | Tên khách sạn | String |

### Target Variables

1. **Sentiment (Classification)**
   - Positive: Rating 4-5 (khách hài lòng)
   - Neutral: Rating 3 (bình thường)
   - Negative: Rating 1-2 (khách không hài lòng)

2. **Aspects (Extraction)**
   - Room: Phòng, giường, phòng tắm
   - Service: Phục vụ, nhân viên
   - Location: Vị trí, khoảng cách
   - Food: Đồ ăn, bữa sáng
   - Price: Giá cả, giá trị
   - Amenities: Tiện nghi (wifi, pool)
   - Cleanliness: Sạch sẽ
   - Noise: Tiếng ồn

---

## 📁 Cấu trúc Project

```
DATA_MINING_PROJECT/
├── README.md                      # Mô tả project
├── requirements.txt              # Dependencies
├── .gitignore                     # Git ignore patterns
├── configs/
│   └── params.yaml               # Tất cả tham số cấu hình
├── data/
│   ├── raw/                      # Dữ liệu thô (chưa xử lý)
│   └── processed/                 # Dữ liệu đã xử lý
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory Data Analysis
│   ├── 02_preprocess_feature.ipynb # Tiền xử lý & Feature Engineering
│   ├── 03_mining_or_clustering.ipynb # Association Rules & Clustering
│   ├── 04_modeling.ipynb         # Classification & Regression
│   ├── 04b_semi_supervised.ipynb  # Semi-supervised Learning
│   └── 05_evaluation_report.ipynb # Đánh giá & Báo cáo
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py              # Load dữ liệu
│   │   └── cleaner.py             # Làm sạch dữ liệu
│   ├── features/
│   │   ├── __init__.py
│   │   └── builder.py             # Trích xuất đặc trưng
│   ├── mining/
│   │   ├── __init__.py
│   │   ├── association.py          # Luật kết hợp
│   │   └── clustering.py          # Phân cụm
│   ├── models/
│   │   ├── __init__.py
│   │   ├── supervised.py           # Classification
│   │   ├── semi_supervised.py      # Semi-supervised
│   │   └── regression.py           # Regression
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py              # Các metric đánh giá
│   │   └── report.py              # Tạo báo cáo
│   └── visualization/
│       ├── __init__.py
│       └── plots.py               # Visualization
├── scripts/
│   └── run_pipeline.py            # Chạy toàn bộ pipeline
├── outputs/
│   ├── figures/                   # Biểu đồ
│   ├── tables/                    # Bảng kết quả
│   ├── models/                    # Model đã train
│   └── reports/                   # Báo cáo
└── app/
    └── streamlit_app.py           # Streamlit Demo App
```

---

## 🛠️ Hướng dẫn cài đặt

### 1. Clone repository

```bash
git clone <repo-url>
cd DATA_MINING_PROJECT
```

### 2. Tạo virtual environment

```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt (Windows)
venv\Scripts\activate

# Kích hoạt (Linux/Mac)
source venv/bin/activate
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK data

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

---

## 🚀 Hướng dẫn sử dụng

### Cách 1: Chạy toàn bộ Pipeline

```bash
python scripts/run_pipeline.py
```

### Cách 2: Chạy từng Notebook

1. Mở Jupyter Notebook:
```bash
jupyter notebook
```

2. Chạy các notebook theo thứ tự:
   - `01_eda.ipynb` - Phân tích dữ liệu
   - `02_preprocess_feature.ipynb` - Tiền xử lý
   - `03_mining_or_clustering.ipynb` - Mining & Clustering
   - `04_modeling.ipynb` - Classification & Regression
   - `04b_semi_supervised.ipynb` - Semi-supervised
   - `05_evaluation_report.ipynb` - Đánh giá

### Cách 3: Chạy Demo App

```bash
cd app
streamlit run streamlit_app.py


---

## ⚙️ Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     PIPELINE OVERVIEW                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────┐      │
│  │ DATA     │───▶│ PREPROCESSING │───▶│ FEATURE ENGINEER │      │
│  │ SOURCE   │    │              │    │                  │      │
│  └──────────┘    └──────────────┘    └────────┬─────────┘      │
│                                               │                 │
│                     ┌─────────────────────────┼─────────────┐   │
│                     ▼                         ▼             ▼   │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌────────┐ │
│  │ ASSOCIATION  │ │ CLUSTERING   │ │CLASSIFICATION│ │REGRES- │ │
│  │ RULES        │ │              │ │              │ │SION    │ │
│  └──────────────┘ └──────────────┘ └──────────────┘ └────────┘ │
│                                          │                      │
│                              ┌────────────┴───────────┐         │
│                              ▼                        ▼         │
│                     ┌──────────────┐         ┌─────────────┐   │
│                     │  SEMI-       │         │ EVALUATION  │   │
│                     │  SUPERVISED  │         │ & REPORT    │   │
│                     └──────────────┘         └─────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Chi tiết từng bước

#### Bước 1: Data Source
- Load dữ liệu từ file CSV
- Validate schema và data types

#### Bước 2: Preprocessing
- Lowercase conversion
- Remove special characters, numbers
- Remove stopwords (NLTK)
- Stemming (SnowballStemmer)
- Handle missing values
- Remove duplicates

#### Bước 3: Feature Engineering
- TF-IDF Vectorization
- Statistical features (word count, length, etc.)
- Aspect extraction keywords

#### Bước 4: Mining & Modeling
- **Association Rules**: Apriori algorithm
- **Clustering**: K-Means, HDBSCAN
- **Classification**: Logistic Regression, Naive Bayes, Random Forest
- **Semi-supervised**: Label Spreading, Self-Training
- **Regression**: Ridge, XGBoost

#### Bước 5: Evaluation
- Metrics: Accuracy, Precision, Recall, F1, MAE, RMSE
- Confusion Matrix
- Learning Curves
- Visualization

---

## 📈 Kết quả

### Các metrics chính

| Model | Metric | Score |
|-------|--------|-------|
| Logistic Regression | F1-macro | ~0.85 |
| Naive Bayes | F1-macro | ~0.82 |
| Random Forest | F1-macro | ~0.88 |
| Label Spreading (20% labels) | F1-macro | ~0.83 |
| XGBoost Regression | RMSE | ~0.65 |

### Biểu đồ chính

- Distribution of Ratings
- Sentiment Distribution
- Word Cloud of Reviews
- Cluster Visualization (t-SNE/PCA)
- Learning Curves (Semi-supervised)
- Confusion Matrix
- Feature Importance

---

## 📚 Dependencies

### Core
- Python 3.8+
- NumPy, Pandas, SciPy

### Machine Learning
- Scikit-learn
- MLxtend (Association Rules)
- XGBoost

### NLP
- NLTK
- BeautifulSoup4
- Regex

### Visualization
- Matplotlib
- Seaborn
- WordCloud

### App
- Streamlit

---

## 👥 License

MIT License - Xem file LICENSE để biết thêm chi tiết.

---

## 📞 Contact

- **Author**: [Your Name]
- **Email**: [your.email@example.com]
- **GitHub**: [github.com/yourusername]

---

**© 2026 - Hotel Reviews Data Mining Project - Đề tài 11**
