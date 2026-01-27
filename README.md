# 💰 Customer Lifetime Value (CLV) Prediction Platform

**Production-ready ML platform for predicting customer lifetime value and segmentation.**

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)

---

## 🎯 Features

### 1. CLV Prediction ✅
- **Multiple ML Models**: Linear Regression & Random Forest
- **Automatic Model Comparison**: R², RMSE, MAE, MAPE metrics
- **Interactive Predictions**: Single customer or batch
- **Export Results**: Download predictions as CSV
- **Intelligent CSV Handler**: Works with ANY format

### 2. Customer Segmentation ✅
- **K-means Clustering**: RFM-based segmentation
- **3D Visualizations**: Interactive customer distribution
- **Marketing Recommendations**: Strategy per segment
- **Export Segments**: Download customer lists

### 3. Cohort Analysis ✅
- **Retention Metrics**: By recency cohorts
- **Revenue Trends**: Visual analytics
- **Actionable Insights**: Data-driven recommendations

### 4. Smart Data Handler ✅
- **Auto-detects columns**: customer_id, revenue, date, frequency
- **Handles transaction data**: Auto-aggregates to customer level
- **Missing data handling**: Intelligent defaults
- **Clear error messages**: Tells you exactly what's wrong

---

## 🚀 Quick Start (5 Minutes)

### Option 1: Try Sample Data (Instant)

1. Visit the live demo (after deployment)
2. Click "🎲 Generate Sample Data"
3. Click "🚀 Train Models"
4. Explore predictions and segmentation!

### Option 2: Upload Your Data

**CSV Format** (any of these work):
```csv
customer_id,revenue,date
CUST_001,500.00,2024-01-15
CUST_002,1200.00,2024-01-10

# OR

customer,sales,purchase_date,frequency
123,500.00,2024-01-15,5
456,1200.00,2024-01-10,3

# OR transaction-level (auto-aggregates)
user_id,amount,timestamp
A,50.00,2024-01-01
A,75.00,2024-01-15
B,100.00,2024-01-10
```

Platform automatically detects and handles all formats!

---

## 📦 Deployment on Render (FREE)

### Step 1: Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: CLV Prediction Platform"
git remote add origin https://github.com/YOUR_USERNAME/clv-platform.git
git push -u origin main
```

### Step 2: Deploy on Render

1. Go to [render.com](https://render.com)
2. Sign up with GitHub (free)
3. Click "New +" → "Web Service"
4. Connect your `clv-platform` repository
5. Configure:
   - **Name**: `clv-prediction-platform`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true`
   - **Instance Type**: **Free**
6. Click "Create Web Service"
7. Wait 3-5 minutes ☕

### Step 3: Your App is Live!

```
🎉 https://clv-prediction-platform.onrender.com
```

**Free tier includes**:
- 750 hours/month (24/7 coverage)
- Automatic HTTPS
- Auto-deploy on git push
- Custom domains supported

---

## 📊 ML Models Explained

### Linear Regression
- **Best for**: Fast predictions, interpretable results
- **Speed**: Very fast
- **Use when**: Linear relationships, need quick training

### Random Forest
- **Best for**: Complex patterns, higher accuracy
- **Speed**: Moderate
- **Use when**: Non-linear relationships, have more data

**Platform automatically compares both and recommends the best!**

---

## 🎓 How to Use

### 1. CLV Prediction

```
Upload Data → Train Models → Compare Performance → Make Predictions → Export
```

**What you get**:
- R² scores for each model
- MAPE (prediction error %)
- Interactive predictions
- Bulk export for all customers

### 2. Customer Segmentation

```
Segment Customers → View 3D Clusters → Get Marketing Strategies → Export Lists
```

**Segments identified**:
- High Value (VIP treatment)
- Medium-High (Upsell opportunities)
- Medium-Low (Engagement campaigns)
- Low Value (Re-activation needed)

### 3. Cohort Analysis

```
Analyze Cohorts → View Revenue Trends → Get Retention Insights
```

**Insights**:
- Which cohorts are most valuable
- Retention patterns
- Re-engagement opportunities

---

## 💡 Use Cases

### Marketing
- **Target high-value customers** for VIP programs
- **Re-engage low-value** customers with promotions
- **Optimize spending** by predicted customer value

### Sales
- **Prioritize leads** by CLV prediction
- **Focus effort** on high-potential customers
- **Upsell strategies** for medium-value segments

### Product
- **Identify features** that drive retention
- **Optimize pricing** based on customer value
- **Personalize experience** per segment

### Finance
- **Revenue forecasting** from customer base
- **Budget allocation** by segment value
- **ROI tracking** on acquisition costs

---

## 📈 Key Metrics

### Model Performance
- **R² Score**: 0-1 (higher better) - variance explained
- **RMSE**: Lower better - average prediction error
- **MAE**: Lower better - absolute error
- **MAPE**: Lower better - percentage error

### Customer Metrics
- **CLV**: Total predicted lifetime value
- **Frequency**: Number of purchases
- **Recency**: Days since last purchase
- **AOV**: Average order value

---

## 🔧 Technical Architecture

```
User Upload → Data Handler → Feature Engineering → ML Models → Predictions
                    ↓
            K-means Clustering → Segmentation → Visualizations
                    ↓
            Cohort Analysis → Retention Metrics → Insights
```

### Components

**Frontend**: Streamlit (interactive UI)
**ML Engine**: Scikit-learn (RF, Linear Regression)
**Visualization**: Plotly (3D charts, interactive)
**Data Processing**: Pandas (intelligent handler)

---

## 📊 Sample Output

### Model Comparison
```
Model              R²      RMSE    MAE     MAPE
Linear Regression  0.82    45.20   32.10   12.5%
Random Forest      0.87    38.50   28.30   10.2%

🏆 Best Model: Random Forest
```

### Segmentation
```
Segment           Count   Avg CLV   Strategy
High Value        250     $1,250    VIP treatment
Medium-High       400     $680      Upselling
Medium-Low        500     $320      Engagement
Low Value         350     $150      Re-activation
```

---

## 🚨 Troubleshooting

### Issue: "Could not detect CUSTOMER ID column"

**Solution**: Ensure your CSV has one of these column names:
- customer_id, customerid, customer, id, user_id, account_id

Or the platform will auto-detect any column with unique values.

### Issue: "Could not detect REVENUE column"

**Solution**: Ensure your CSV has one of these column names:
- revenue, sales, amount, value, purchase, spent

Or any positive numeric column will be detected.

### Issue: Build fails on Render

**Solution**: Check `render.yaml` has correct Python version:
```yaml
envVars:
  - key: PYTHON_VERSION
    value: 3.9.18
```

---

## 📁 Project Structure

```
clv-platform/
├── app.py                  # Main application (650 lines)
├── src/
│   └── data_handler.py     # Intelligent CSV handler (280 lines)
├── requirements.txt        # Dependencies
├── render.yaml            # Render config
├── .streamlit/
│   └── config.toml        # Streamlit config
├── README.md              # This file
└── DEPLOY.md              # Deployment guide
```

---

## 🎯 What Makes This Production-Ready

✅ **Complete Functionality**: No "Coming Soon" placeholders
✅ **Error Handling**: Graceful failures with clear messages
✅ **Smart Data Handler**: Works with ANY CSV format
✅ **Multiple Models**: Comparison and automatic selection
✅ **Export Capabilities**: Download all results
✅ **Production Deployment**: Render config included
✅ **Documentation**: Complete guides and examples
✅ **Tested**: All features working end-to-end

---

## 📧 Support

**Built by**: Vaibhav Sathe  
**Email**: vaibhavag0207@gmail.com  
**LinkedIn**: [vaibhav-sathe-115507194](https://www.linkedin.com/in/vaibhav-sathe-115507194)

---

## 📄 License

MIT License - Free to use and modify

---

**🚀 Ready to deploy! All features work completely. No demos, no placeholders.**
