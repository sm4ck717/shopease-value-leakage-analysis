# 🛍️ ShopEase – Diagnosing Value Leakage & Optimizing Growth

> **Unravel 6.0 – Case Competition**
> Team **Green Analysts** · IISWBM, Kolkata (University of Calcutta)
> Case hosted by IIM Sambalpur

A data-backed diagnosis of **what is breaking** at ShopEase, a fashion e-commerce business, and what to do about it. The project covers exploratory analysis, root cause analysis, RFM customer segmentation, BCG-style portfolio analysis, a **return-probability prediction model**, and an interactive **Streamlit** app with optional GenAI insights.

---

## 📌 Business Problem

ShopEase is growing, but early warning signs are showing: rising complaints, inconsistent sales across product categories, and declining marketing effectiveness. Leadership lacks visibility into **which customers, products, channels or operational processes** are driving these issues.

**Objective:** use internal data to pinpoint root causes before revenue, margins and brand trust are affected, and recommend corrective, scalable actions.

---

## 🧠 Approach

1. **Exploratory Data Analysis:** overall health, customers, products, marketing and operations
2. **Customer Segmentation:** RFM analysis (8 segments)
3. **Root Cause Analysis:** four problem areas, each backed by evidence from the data
4. **Product & Brand Portfolio Analysis:** BCG-matrix framing
5. **Predictive Modelling:** return-probability classifier to flag risky orders *before* fulfilment
6. **Strategy & Recommendations:** customer, product and operational levers

---

## 🗂️ Dataset

`shop_ease_fashion_dataset.csv` has **10,000 orders × 21 columns**, covering:

| Area | Examples |
|---|---|
| Customers | Age group (18–24, 25–34, 35–44, 45+), city tier, tenure |
| Acquisition | Email, Google, Influencers, Instagram, Referrals, plus channel cost |
| Products | 6 categories (Accessories, Active Wear, Casuals, Ethnic, Formals, Lounge Wear), 24 sub-categories, brands A–F |
| Operations | Delivery time, refund amounts, cancellation reasons |
| Payments | Card, COD, Net Banking, UPI, Wallet |

All data is for calendar year 2025.

---

## 📊 Key Findings

### Overall health (2025)

- About **₹18.4M revenue** across 10,000 orders, with an average order value of about **₹1.84K**
- **369 complaints** and **688 returned orders** in total (about 3.7% and 6.9% of orders)
- Complaints peak in **October**, which is also when AOV dips to its lowest point (about ₹1.56K)

### Root causes

| Problem | Evidence |
|---|---|
| **Product expectation mismatch** | Accessories have the highest return rate (9.22%), followed by Ethnic (8.98%). Watches, Lehengas and Ethnic Sets exceed 10% return rate. Brand E returns most (7.89%). |
| **Delivery delays** | Complaint rate is **2.04%** for 0–3 day deliveries but **25.69%** for 7+ days. Average refund value is highest for 7+ days (~₹251). Tier-3 cities show **9.01%** complaint rate vs 2.20% in Tier-1. |
| **Low-quality customer acquisition** | Instagram drives the most volume (28.3%) but has the highest complaint rate (4.17%). Influencer has the highest CAC (~₹95) without a higher AOV. Referral is the cheapest (~₹5.5 CAC) with the lowest complaints (3.01%). |
| **Weak retention** | At Risk and Lost customers form roughly a third of the base. Loyal customers still show a ~4.2% complaint rate. Recent customers have high return rates (~7.5%). |

### Other operational signals

- **COD** orders have the worst profile: **12.49%** return rate and **6.33%** complaint rate, vs about 4–6% returns for prepaid modes
- **Payment failure** is the largest cancellation reason (~30% of cancellations)

### RFM segments

Customers are scored on Recency, Frequency and Monetary value and grouped into: Top Customers, Loyal Customers, Recent Customers, Big Spenders, Potential Loyalists, At Risk, Lost, Low Value and Others. Loyal Customers account for the largest share of orders (~38%).

---

## 🤖 Return Probability Model

Predicts whether an order is likely to be returned, so risky orders can be flagged before dispatch.

| Item | Detail |
|---|---|
| Model | **Logistic Regression** (selected after benchmarking multiple classifiers) |
| Class imbalance | **SMOTE** on training data only (~7% of orders are returns) |
| Threshold | Tuned to **0.30** to prioritize recall over accuracy |
| Accuracy | ~65% |
| Recall (returns) | ~47% |
| Recall (not returned) | ~67% |
| Business check | Actual return rate is **9.50%** in the high-risk group vs **5.55%** in the low-risk group |

**Features:** age group, city tier, tenure, product category, sub-category, brand, discount %, acquisition channel, channel cost, delivery time and payment mode.

**Why Logistic Regression?** It is interpretable and stable, it gives the best recall after threshold tuning, and its drivers map directly onto operational and customer interventions.

**Drivers that increase return risk:** COD, Ethnic and Accessories categories, Pajama Sets and Watches, longer delivery times.
**Drivers that reduce return risk:** prepaid payment modes, Referral and Google channels, Formal and Casual categories, older and repeat customers.

---

## 💡 Recommendations

**Customer strategy**
- Replace one-size-fits-all CX with **RFM-based service tiers**
- Proactive retention offers for At Risk customers, low-cost automated engagement for Lost and Low Value customers
- Improve first-purchase CX: clear delivery timelines, sizing guidance, restricted deep discounts for first-time buyers in high-return categories, a feedback loop within 48 hours

**Product portfolio (BCG framing)**
- **Stars (Formals, Casuals):** invest with strict QC and faster SLAs
- **Cash cow (Ethnic):** protect the revenue base by fixing sizing and brand controls
- **Question mark (Accessories):** fix quality and SKU mix before scaling
- **Dogs (Active Wear, Lounge Wear):** deprioritize, cut SKUs, limit spend

**Operations**
- Separate **delivery SLAs for Tier-1, Tier-2 and Tier-3** cities, and reduce 7+ day deliveries
- Pre-dispatch cancellation for delayed orders, with store credit or partial refunds in place of full refunds where feasible
- Restrict COD for high-risk customers and categories, incentivize prepaid modes, and improve payment retry flows
- Flag high-risk orders with the model for proactive intervention

---

## 🗃️ Repository Structure

| File | Description |
|---|---|
| `analysis.ipynb` | EDA, root cause analysis and RFM segmentation |
| `ml.ipynb` | Model benchmarking |
| `ml_log_reg.ipynb` | Final Logistic Regression model with SMOTE and threshold tuning |
| `app.py` | Streamlit **Return Probability Engine** |
| `shop_ease_fashion_dataset.csv` | Dataset (10,000 × 21) |
| `unravel_dash.pbix` | Power BI dashboard: *ShopEase Performance Analytics & Value Leakage* |
| `Unravel 6.0(Case).pdf` | Official case statement |
| `Green Analysts_IISWBM Kolkata.pdf` | Final presentation deck |
| `requirements.txt` | Python dependencies |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- (Optional) A [Google Gemini API key](https://aistudio.google.com/app/apikey) for AI-generated insights

### Installation

```bash
git clone https://github.com/sm4ck717/shopease-value-leakage-analysis.git
cd shopease-value-leakage-analysis

# (optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Run the Streamlit app

```bash
streamlit run app.py
```

Run it from the repository root so the app can find `shop_ease_fashion_dataset.csv`. The model trains when the app starts and is cached afterwards.

### Explore the notebooks

```bash
jupyter notebook
```

Open `analysis.ipynb` first, then `ml.ipynb` and `ml_log_reg.ipynb`.

### Power BI dashboard

Open `unravel_dash.pbix` in [Power BI Desktop](https://powerbi.microsoft.com/desktop/).

---

## 🖥️ App Features (`app.py`)

- **Prediction tab:** enter order details in the sidebar (customer profile, product, payment, channel, discount, delivery days) and get a return probability, a High/Low risk label at the 0.30 threshold, and a risk meter
- **GenAI Analyst:** paste a Gemini API key in the sidebar to get a short explanation of why the order is risky and three suggested actions to prevent the return. The key is used only for the current session and is not stored.
- **Model Insights tab:** top 15 drivers of returns from the model coefficients (red increases risk, green reduces it)
- **Dataset tab:** preview of the first 100 rows

---

## 🛠️ Tech Stack

Python · Pandas · NumPy · Matplotlib · Seaborn · scikit-learn · imbalanced-learn (SMOTE) · Streamlit · Google Gemini API · Jupyter Notebook · Power BI

---

## 👥 Team – Green Analysts

- **Shreyan Sarkar**
- **Subhodip Manna**

Indian Institute of Social Welfare & Business Management (IISWBM), Kolkata, affiliated to the University of Calcutta.

---

## 📄 License

No license has been specified yet. Consider adding one (for example MIT) if you want others to reuse this work.
