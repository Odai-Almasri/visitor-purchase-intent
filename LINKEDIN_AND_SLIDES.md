# LinkedIn post (replace [link] and [GitHub link]; add your teammates' tags: none, you worked solo)

Most online shoppers leave without buying, and stores usually find out too late.

In my NextGen Institute ML project, I built an app that scores a browsing session and tells the store what to do: show a coupon, open a chat assistant, or do nothing, so discounts aren't wasted on people who would buy anyway.

• Trained and tuned 3 models (Logistic Regression, Random Forest, Gradient Boosting) on 12,205 real shopping sessions
• Best model: ROC-AUC 0.937, catching 72% of real buyers; the top 10% of scored sessions held 51% of all buyers
• Biggest lesson: only 15.6% of sessions convert, so accuracy was misleading. A model that never predicts a purchase is 84.4% "accurate" and useless. Recall and PR-AUC told the real story.

Try the live app: [link] · Code: [GitHub link]

#MachineLearning #DataScience #Ecommerce #NextGenInstitute

---

# CV bullet
Developed a purchase-intent model on 12,205 e-commerce sessions (ROC-AUC 0.937) and deployed a Streamlit app that recommends real-time actions (chat / coupon / none) to convert visitors. Tools: Python, scikit-learn, Streamlit.

---

# Slides (10-12 slides, ~10 minutes)
1. **Title:** project name, your name, NextGen Institute
2. **Problem:** most visitors don't buy (15.6% convert); discounts for everyone waste margin
3. **Decision & metric:** no action / coupon / chat; why ROC-AUC, PR-AUC, recall (not accuracy)
4. **Data:** UCI source, 12,205 sessions, 17 features, class balance chart
5. **EDA insight 1-2:** conversion by visitor type (new 24.9% vs returning 14.1%) and by month (Nov 25.5%, Feb 1.7%)
6. **EDA insight 3-4:** PageValues (mean 27.27 vs 2.00) and ExitRates (0.020 vs 0.046), correlation heatmap
7. **Pipeline:** split first, Pipeline diagram, no leakage
8. **Models & tuning:** baseline + 3 models, 5-fold CV, RandomizedSearchCV
9. **Results:** results table (baseline vs best), ROC/PR curves
10. **Threshold policy & error analysis:** thresholds 0.36 / 0.095, confusion matrix, misses when PageValues = 0, lift table (top 10% = 51% of buyers)
11. **Business recommendation + limitations:** the policy, A/B test next step, PageValues caveat
12. **Live demo:** app screenshot/QR + GitHub link
