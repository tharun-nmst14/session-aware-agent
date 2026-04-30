import streamlit as st
import os
from groq import Groq

# ========== CONFIG ==========
MODEL_NAME = "llama-3.1-8b-instant"
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ========== SESSION KNOWLEDGE ==========
SESSION_KNOWLEDGE = [
    {
        "session": "Cloud Fundamentals",
        "content": "Cloud computing, IaaS, PaaS, SaaS, Azure, AWS, GCP, REST APIs."
    },
    {
        "session": "Azure Fundamentals",
        "content": "Azure regions, availability zones, subscriptions, VMs, storage, pricing."
    },
    {
        "session": "Azure Migration",
        "content": "Azure Migrate, IaaS vs PaaS, RBAC roles, cutover strategies."
    },
    {
        "session": "DevOps & Terraform",
        "content": "DevOps, Azure DevOps, Terraform workflow, IaC, YAML pipelines."
    },
    {
        "session": "IoT, Containers & AKS",
        "content": "IoT Hub, Docker, Kubernetes, AKS, YAML/JSON, Grafana."
    }
]

# ========== FUNCTIONS ==========
def is_session_meta_question(question):
    keywords = ["entire session", "all sessions", "analyze sessions", "summarize sessions"]
    return any(k in question.lower() for k in keywords)

def check_scope(question):
    prompt = f"""
Check if the question is related to Cloud, Azure, DevOps, Terraform, IoT, AKS.
Answer ONLY YES or NO.

Question: {question}
"""
    res = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return "YES" in res.choices[0].message.content.upper()

def retrieve_knowledge():
    return "\n".join([s["content"] for s in SESSION_KNOWLEDGE])

def answer_from_sessions(question):
    prompt = f"""
Answer ONLY using the following session knowledge.
If not found, say "Not covered in sessions".

Knowledge:
{retrieve_knowledge()}

Question:
{question}
"""
    res = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return res.choices[0].message.content

def general_answer(question):
    res = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": question}],
        temperature=0.7
    )
    return res.choices[0].message.content

# ========== STREAMLIT UI ==========
st.set_page_config(page_title="Session-Aware Agent", layout="centered")

st.title("Session‑Aware Agentic AI")
st.write("Answers from session knowledge first, otherwise uses general AI.")

question = st.text_input("Ask a question:")

if st.button("Ask"):
    if question:
        with st.spinner("Agent thinking..."):
            if is_session_meta_question(question):
                answer = answer_from_sessions("Summarize all sessions")
                st.success(answer)

            elif check_scope(question):
                answer = answer_from_sessions(question)
                st.success(answer)

            else:
                answer = general_answer(question)
                st.info(answer)
    else:
        st.warning("Please enter a question.")
