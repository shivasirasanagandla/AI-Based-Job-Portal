# 🚀 AI-Powered Job Portal

An intelligent, research-driven recruitment platform that transforms traditional job search and hiring with AI-powered resume screening, personalized job recommendations, and real-time interview simulations.

## 📌 Overview
This project is a comprehensive AI-powered job portal designed to bridge the gap between job seekers and recruiters. Built with cutting-edge technologies like BERT, Graph Neural Networks, SHAP explainability, and real-time NLP pipelines, the system automates and enhances end-to-end recruitment workflows.

## 🧠 Key Features

- **🔍 Resume Screening Module**
  - Parses PDF/DOCX resumes using PyMuPDF
  - Classifies candidate job domain using a fine-tuned BERT model
  - Supports explainability via SHAP (planned)

- **💼 Hybrid Job Recommendation Engine**
  - Combines TF-IDF content-based filtering and collaborative filtering
  - Cold-start mitigation using GNNs (planned)
  - Delivers personalized job suggestions in real time

- **🎤 Interview Simulation**
  - Uses speech-to-text NLP and keyword analysis
  - Analyzes sentiment and fluency
  - Plans integration of facial emotion detection (MediaPipe, OpenCV)

- **🖥️ Frontend UI/UX**
  - Built with React + Vite for high-speed, mobile-responsive interaction
  - Role-based dashboards for Job Seekers, Recruiters, and Trainers/Admins

## 🧱 Tech Stack

- **Frontend**: React, Vite, Tailwind CSS  
- **Backend**: FastAPI, Python  
- **ML/NLP**: PyTorch, HuggingFace Transformers, Scikit-learn  
- **Data**: PyMuPDF, Pandas, SQLite/PostgreSQL  
- **AI Explainability**: SHAP (planned), Fairlearn  
- **Realtime/Streaming**: WebRTC, Web Speech API (planned)  
- **DevOps**: Docker (optional), GitHub Actions (for CI/CD)

## 📁 Project Structure

```bash
├── backend/
│   ├── app/
│   ├── models/
│   └── resume_screening.py
├── frontend/
│   ├── src/
│   └── App.tsx
├── data/
│   └── sample_resumes/
├── notebooks/
│   └── Resume_Screening.ipynb
├── diagrams/
│   └── architecture.png
└── README.md
```

## 📚 Research Backing

This project is informed by academic and enterprise research including:

- Resume classification using BERT  
- GNN-based job-user modeling  
- SHAP for explainable hiring  
- Emotion detection in interviews  
(See `docs/Research_References.md` for full citations.)

## 📌 Getting Started

1. **Clone the repo**  
```bash
git clone https://github.com/shivasirasanagandla/AI-Based-Job-Portal.git
cd ai-job-portal
```

## 🛡️ License

This project is open-source under the MIT License. See [LICENSE](LICENSE) for details.
