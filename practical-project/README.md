# 🎯 Complete Customer Segmentation Project

An end-to-end machine learning project demonstrating customer segmentation for targeted marketing.

## 🎯 Project Objective

Segment customers into distinct groups based on their purchasing behavior and demographics to enable:
- Personalized marketing campaigns
- Product recommendations
- Customer retention strategies

## 📊 Dataset

- **Source**: Mall Customers Dataset
- **Samples**: 200 customers
- **Features**: 
  - CustomerID
  - Gender
  - Age
  - Annual Income (k$)
  - Spending Score (1-100)

## 🛠️ Methodology

1. **Data Exploration** - Understand distributions and relationships
2. **Preprocessing** - Handle missing values, scale features
3. **Clustering** - Apply K-Means with optimal k selection
4. **Analysis** - Interpret cluster characteristics
5. **Visualization** - Create intuitive visualizations
6. **Business Recommendations** - Actionable insights

## 📈 Results

### Customer Segments Identified

| Segment | Name | Income | Spending | Size | Strategy |
|---------|------|--------|----------|------|----------|
| 0 | High Income, Low Spending | High | Low | 39 | Premium products, loyalty programs |
| 1 | Average Income, Average Spending | Medium | Medium | 81 | General marketing, bundle offers |
| 2 | High Income, High Spending | High | High | 35 | Exclusive offers, VIP treatment |
| 3 | Low Income, High Spending | Low | High | 22 | Installment plans, targeted discounts |
| 4 | Low Income, Low Spending | Low | Low | 23 | Value products, engagement campaigns |

## 💡 Business Recommendations

1. **Segment 2 (Premium Shoppers)**: Launch exclusive membership program
2. **Segment 3 (Aspirational)**: Offer flexible payment options
3. **Segment 0 (Cautious Spenders)**: Build trust with quality guarantees
4. **Segment 4 (Budget Shoppers)**: Focus on value products
5. **Segment 1 (Average Customers)**: Use as control group for A/B testing

## 🚀 How to Run

1. Navigate to this directory
2. Open the notebook:
```bash
jupyter notebook notebooks/complete_customer_analysis.ipynb
