# 🧠 Smart Product Pricing Challenge — Amazon ML Challenge 2025

![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Regression-blue)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)
![License](https://img.shields.io/badge/License-MIT-green)
![Dataset](https://img.shields.io/badge/Dataset-Provided-orange)

---

## 📌 Overview

E-commerce pricing is a critical factor influencing customer behavior and marketplace success.  
In this challenge, our goal is to develop a **Machine Learning model** capable of predicting the **optimal product price** based on both **textual** and **visual** product details.

The challenge tests participants’ ability to handle **multimodal data** (text + images), perform **feature engineering**, and design **robust ML pipelines** that generalize well on unseen data.

---

## 🧾 Problem Statement

Build a model that predicts the price of a product given its **catalog content** (title, description, and item quantity) and **image**.

### Input Columns:
| Column | Description |
|---------|-------------|
| `sample_id` | Unique identifier for each record |
| `catalog_content` | Concatenation of title, description, and Item Pack Quantity (IPQ) |
| `image_link` | Public URL to product image |
| `price` | Target variable (available only in training data) |

---

## 📂 Dataset Details

| File | Description |
|------|-------------|
| `dataset/train.csv` | Training data with prices |
| `dataset/test.csv` | Test data (without prices) |
| `dataset/sample_test.csv` | Sample test input file |
| `dataset/sample_test_out.csv` | Example output format |

**Dataset Size:**
- **Train:** 75,000 products  
- **Test:** 75,000 products  

---

## 🎯 Objective

Develop a model that accurately predicts the **product price** for each entry in the test set.

Output must be a CSV file with the following structure:

```csv
sample_id,price
00001,499.99
00002,349.50
00003,1299.00
...
