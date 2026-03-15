import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch

# ----------------------------
# 🎨 App Configuration
# ----------------------------
st.set_page_config(page_title="AI Career Path Recommender", page_icon="🎯", layout="wide")

st.markdown("""
    <style>
        .main {
            background-color: #f9f9f9;
        }
        .career-card {
            background-color: white;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        .career-card:hover {
            transform: scale(1.02);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# 🎯 Page Header
# ----------------------------
st.title("💼 AI Career Path Recommender")
st.markdown("Find your ideal career path based on your **skills and interests** 🚀")

# ----------------------------
# ⚙️ Model Setup
# ----------------------------
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# ----------------------------
# 🧩 Career Knowledge Base (no dataset needed)
# ----------------------------
career_data = {
    "Data Scientist": "Python, Machine Learning, Statistics, SQL, Data Analysis",
    "Software Engineer": "Programming, Algorithms, Data Structures, Java, Python, C++",
    "AI Engineer": "Deep Learning, Python, TensorFlow, PyTorch, NLP, Computer Vision",
    "Web Developer": "HTML, CSS, JavaScript, React, Node.js, Frontend, Backend",
    "Mobile App Developer": "Flutter, Kotlin, Swift, Android, iOS, UI/UX",
    "Cloud Engineer": "AWS, Azure, DevOps, Docker, Kubernetes, Cloud Architecture",
    "Cybersecurity Analyst": "Network Security, Firewalls, Ethical Hacking, Risk Analysis",
    "Data Analyst": "Excel, Power BI, SQL, Python, Data Visualization, Analytics",
    "Game Developer": "Unity, Unreal Engine, C#, 3D Modeling, Animation",
    "DevOps Engineer": "CI/CD, Docker, Jenkins, Kubernetes, Linux, Automation",
    "UI/UX Designer": "Figma, Adobe XD, Prototyping, Wireframing, Design Thinking",
    "Blockchain Developer": "Solidity, Ethereum, Smart Contracts, Cryptography",
    "Product Manager": "Business Analysis, Agile, Communication, Strategy, Leadership",
    "Cloud Architect": "AWS, Azure, Infrastructure, Scalability, Networking",
    "Research Scientist": "AI, Mathematics, NLP, Research, Problem Solving"
}

# ----------------------------
# 🧠 Input Section
# ----------------------------
branch = st.selectbox("🎓 Select your branch:", [
    "Computer Science", "Information Technology", "Electronics", "Mechanical", "Civil", "Electrical", "Other"
])

skills = st.text_area("💡 Enter your skills or interests (comma separated):", placeholder="e.g. Python, Java, AI, Web Development")

# ----------------------------
# 🚀 Recommendation Logic
# ----------------------------
if st.button("Get Career Recommendations 🚀"):
    if skills.strip() == "":
        st.warning("Please enter some skills or interests first.")
    else:
        with st.spinner("Analyzing your skills... 🔍"):
            # Encode user input and careers
            user_embedding = model.encode(skills, convert_to_tensor=True)
            career_names = list(career_data.keys())
            career_embeddings = model.encode(list(career_data.values()), convert_to_tensor=True)

            # Compute similarity scores
            scores = util.cos_sim(user_embedding, career_embeddings)[0]

            # Sort top results
            top_results = torch.topk(scores, k=5)

            st.subheader("✅ Top Career Recommendations:")

            for idx in top_results.indices:
                career = career_names[idx]
                score = float(scores[idx] * 100)  # convert float32 → float
                skills_needed = career_data[career]

                with st.container():
                    st.markdown(f"""
                        <div class="career-card">
                            <h3>🎯 {career}</h3>
                            <p><b>Match Score:</b> {score:.2f}%</p>
                            <p><b>Recommended Skills to Learn:</b> {skills_needed}</p>
                        </div>
                    """, unsafe_allow_html=True)

                    st.progress(float(score) / 100)

        st.success("✅ Recommendations generated successfully!")

# ----------------------------
# 🧾 Footer
# ----------------------------
st.markdown("""
---
👩‍💻 **Developed by:** Varsha's AI Project  
💬 Powered by Sentence Transformers & Streamlit  
🌟 _Find your future, today._
""")
