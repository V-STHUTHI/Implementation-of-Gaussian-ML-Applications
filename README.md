<img width="2533" height="1689" alt="image" src="https://github.com/user-attachments/assets/4d7617fb-ac22-4c41-aa15-26d5da82d99b" />🔢 Implementation: Applications of Normal Distribution in ML
This repository is a Python implementation of the research paper:
"Some applications of Normal distribution in Machine Learning algorithms" Published in: International Journal of Engineering Inventions (February 2025)
🔬 Research Context:
As highlighted in the research by Tran Thi Bich Hoa, probability theory is the foundation of modern Machine Learning. 
This project implements the paper's core objective: studying the applications of normal distribution and using Python to verify data normality before modeling.
🛠️ Key Implementations
1. Data Normalization ("Whitening")
2. Following the paper's focus on the "whitening" technique , this project standardizes input data to a mean of 0 and a standard deviation of 1 using the formula:
3. y = \frac{x - \text{mean}}{\text{standard\_deviation}}
4. This ensures that data components with different value domains are processed effectively by algorithms like Linear and Logistic Regression.
5. 2. The Empirical Rule & Anomaly DetectionThe project utilizes the Empirical Rule described in the paper to identify outliers:
   3. 68.26% of data within 1sigma
   4. 95.44% of data within 2sigma
   5. 99.7% of data within 3sigma Our algos/anomaly.py module uses the \mu \pm 3\sigma threshold to detect rare events or outliers
Dimensionality Reduction (PCA):
  As suggested in the paper, we implement Principal Component Analysis (PCA) to find the directions of maximum variance.
  This simplifies complex data like MNIST while maintaining the core statistical distribution.
Clustering with GMM:
  We implement Gaussian Mixture Models (GMMs), which the research identifies as a key application for modeling complex data distributions and image segmentation.
 Project Structure
algos/: Implementations of GMM, PCA, Linear, and Logistic Regression.
prep/: Normalization logic using the "Whitening" technique.
viz/: Visualizations for checking data normality via histograms and bell curves.
frontend/: A Streamlit dashboard to interact with the research visualisations.
How to Run
  Install dependencies: pip install -r requirements.txt
  Launch the Workbench: streamlit run frontend/main.py
