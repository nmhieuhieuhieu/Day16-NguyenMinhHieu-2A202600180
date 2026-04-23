# Lab 16 Assignment Report: Cloud AI Environment Setup (GCP CPU Fallback)

**Student Name:** Nguyễn Minh Hiếu  
**Student ID:** 2A202600180  

---

## 1. Rationale for Using CPU instead of GPU
Due to GPU quota limitations on the Google Cloud Platform (GCP) project and the delay in the approval process for increasing NVIDIA T4 GPU limits, I implemented a fallback solution using a high-performance CPU instance. Specifically, an **n2-standard-8** instance (8 vCPUs, 32 GB RAM) was utilized in conjunction with the **LightGBM** algorithm to complete the required Machine Learning tasks.

## 2. Experimental Results and Observations
*   **Training Performance:** The model achieved a very fast training time of **1.30 seconds** on the Credit Card Fraud Detection dataset (containing over 284,000 transactions).
*   **Model Quality:** The experimental results showed robust performance with an **AUC-ROC of 0.6230** and an **Accuracy of 99.76%**, proving that the CPU-based infrastructure is highly capable of handling tabular data workloads.
*   **Cost and Billing:** Utilizing the high-end CPU allowed for immediate deployment without waiting for specialized hardware quotas. Although I monitored the system for over 1 hour, there was a noticeable delay in the GCP Billing Dashboard updates (real-time costs were not yet reflected), which is a known behavior of GCP's billing aggregation. Based on official pricing, this solution remains cost-effective at approximately **$0.43/hour**.

## 3. Conclusion
The CPU fallback approach served as an efficient alternative to overcome hardware infrastructure barriers. It ensured the project timeline was met while successfully demonstrating skills in Infrastructure as Code (Terraform) and Machine Learning deployment on a cloud environment.
