# Customer Behavior Intelligence System

A machine learning based customer segmentation and analytics system built using RFM (Recency, Frequency, Monetary) Analysis, K-Means Clustering, and Flask.

The system analyzes customer transaction behavior, automatically segments customers into meaningful groups, and provides business insights and retention recommendations through an interactive web interface.

---

## Features

- Customer segmentation using RFM Analysis
- K-Means clustering for behavioral grouping
- Data cleaning and preprocessing pipeline
- Outlier detection and removal using IQR method
- StandardScaler based feature normalization
- Dynamic customer classification into:
  - High Value Customers
  - Regular Customers
  - Low Value Customers
- Interactive Flask dashboard
- Customer lookup and segmentation insights
- Automated visualization generation using Seaborn and Matplotlib

---

## Tech Stack

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Pickle

---

## Dataset

**Online Retail Dataset**

- 541,000+ retail transactions
- 4,372 unique customers
- Customer purchase history used for RFM analysis and segmentation

---

## Methodology

### 1. Data Preprocessing

- Removed missing values
- Created Monetary (Amount) feature
- Converted transaction dates to datetime format
- Generated RFM metrics

### 2. Outlier Removal

Applied IQR-based filtering on:

- Amount
- Frequency
- Recency

### 3. Feature Scaling

Used StandardScaler to normalize RFM features before clustering.

### 4. Customer Segmentation

Evaluated K-Means, DBSCAN, and Agglomerative Clustering using Silhouette Score and selected K-Means for customer segmentation.

### 5. Business Intelligence Layer

Customers are automatically categorized as:

- High Value Customers
- Regular Customers
- Low Value Customers

The system generates retention and engagement recommendations for each segment.

---

## Visualizations

The application generates:

- Customer Segments vs Amount
- Customer Segments vs Frequency
- Customer Segments vs Recency
- Customer Segment Distribution (Donut Chart)

---

## Project Structure

```text
Customer-Behavior-Intelligence-System/
│
├── app.py
├── model.pkl
├── customer-behavior-intelligence.ipynb
│
├── templates/
│   └── index.html
│
├── static/
│   ├── cluster_donut.png
│   ├── ClusterId_Amount.png
│   ├── ClusterId_Frequency.png
│   └── ClusterId_Recency.png
│
└── README.md
```

---

## Key Outcomes

- Processed and analyzed 541K+ retail transactions
- Generated RFM-based customer profiles
- Built an end-to-end machine learning pipeline
- Developed an interactive Flask dashboard for customer analytics
- Enabled segment-based business recommendations for customer retention and engagement

---

## Author

**Tanya Gupta**

MCA, IGDTUW  
Machine Learning • Data Analytics • Software Development
