https://docs.google.com/presentation/d/1OfjhUqugAeKv6Un1PFt8VCHHkdXIF5mgDMtxz5Ef3s0/edit?usp=sharing
# creditcardFraud
## over view
區分信用卡交易是否為盜刷，避免持卡者被錯誤扣款 但其中總有模稜兩可的狀況，錯誤的認為正確的持卡者為盜刷，將會造成客戶體驗不佳甚至流失的問題，因此在追求準確預測之餘，也需考量兩種型態錯誤造成的成本，以成本與錯誤機率的綜合作為閥值的考量依據。 
資料集來源：信用卡資料盜刷交易
資料描述：
由於保護消費者的個資，並未提供原始交易資料，提供資料是經過PCA降維後的結果。
## Highly imbalanced
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions. 
## Performance Metrics  
F1 Score for Fraud Accuracy is misleading!
所以我們使用Cost-sensitive
定義含有k個樣本的
Class y 之權重= N_sample / (N_class * k)
樣本數越少的class權重越高

可以自由使用http://contrib.scikit-learn.org/imbalanced-learn/stable/index.html中處理資料的方式
測試各種分類器的表現

