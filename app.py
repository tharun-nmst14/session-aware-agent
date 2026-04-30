import streamlit as st
import os
from groq import Groq

# ================= CONFIG =================
MODEL_NAME = "llama-3.1-8b-instant"
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ================= SESSION KNOWLEDGE (REAL MoMs) =================

SESSION_KNOWLEDGE = [
    {
        "session": "Session 1 – Cloud Fundamentals",
        "content": """
This session introduced cloud computing fundamentals and the transition
from on-premises data centers to cloud platforms. It covered the benefits
of cloud computing such as scalability, global reach, and cost efficiency.
Service models IaaS, PaaS, and SaaS were explained with examples.
Public, private, and hybrid cloud deployment models were discussed.
Major cloud providers Azure, AWS, and GCP were compared.
REST, SOAP, and WebSockets API technologies were explained.
"""
    },
    {
        "session": "Session 2 – Azure Fundamentals",
        "content": """
This session focused on Microsoft Azure core concepts such as regions,
availability zones, subscriptions, and resource groups.
Virtual machine creation was discussed including compute, storage,
and networking components. Azure VM series such as B, D, E, F, M, and N
were explained along with disk types like HDD, SSD, and Premium SSD.
Pay-as-you-go pricing and VM lifecycle operations were covered.
"""
    },
    {
        "session": "Session 3 – Cloud Migration",
        "content": """
This session covered migration from on-premises infrastructure to Azure.
Differences between IaaS and PaaS were explained in migration scenarios.
Azure Migrate tool was introduced along with RBAC roles such as Owner,
Contributor, and Reader. Cutover strategies including test, live, and post
cutover were discussed. Practical deployment using App Service and Azure
SQL Database was demonstrated.
"""
    },
    {
        "session": "Session 4 – DevOps & Terraform",
        "content": """
This session introduced DevOps and DevSecOps concepts.
Azure DevOps services such as Repos and Pipelines were explained.
Terraform was discussed as an Infrastructure as Code tool along with its
advantages and workflow commands like init, plan, apply, and destroy.
YAML pipelines and agent pools were covered for CI/CD automation.
"""
    },
    {
        "session": "Session 5 – IoT, Containers & AKS",
        "content": """
This session focused on cloud-native technologies.
Azure IoT Hub and device-to-cloud and cloud-to-device communication
were explained. Docker containers and Kubernetes orchestration concepts
were discussed. Azure Kubernetes Service (AKS), YAML and JSON configuration,
Grafana monitoring, tagging, and container registries were covered.
"""
    }
]

# ================= AGENT FUNCTIONS =================

def is_session_meta_question(question: str) -> bool:
    keywords = [
        "overview of sessions",
        "analyze my sessions",
        "summarize sessions",
        "entire sessions",
        "what did we cover",
        "session summary"
    ]
    q = question.lower()
    return any(k in q for k in keywords)


def check_scope(question: str) -> bool:
    prompt = f"""
Determine whether the question is related to any of the following topics:
Cloud Computing, Microsoft Azure, IaaS, PaaS, SaaS, Cloud Migration,
DevOps, Terraform, IoT, Docker, Kubernetes, AKS.

Respond ONLY with YES or NO.

Question:
{question}
"""
    res = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return "YES" in res.choices[0].message.content.upper()


def retrieve_relevant_knowledge(question: str) -> str:
    matched = []
    q_words = question.lower().split()

    for s in SESSION_KNOWLEDGE:
        if any(word in s["content"].lower() for word in q_words):
            matched.append(f"{s['session']}:\n{s['content']}")

    # fallback to all sessions if no specific match
    if not matched:
        matched = [f"{s['session']}:\n{s['content']}" for s in SESSION_KNOWLEDGE]

    return "\n\n".join(matched)


def answer_from_sessions(question: str, knowledge: str) -> str:
    prompt = f"""
You are an AI assistant.
Answer the question STRICTLY using the session knowledge below.
If the answer is not present, say: "This was not covered in the sessions."

Session Knowledge:
{knowledge}

Question:
{question}
"""
    res = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return res.choices[0].message.content.strip()


def general_answer(question: str) -> str:
    res = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": question}],
        temperature=0.7
    )
    return res.choices[0].message.content.strip()

# ================= STREAMLIT UI =================

st.set_page_config(page_title="Session‑Aware Agentic AI", layout="centered")

st.title(" Session‑Aware Agentic AI")
st.write(
    "This agent answers questions using **training session knowledge first**. "
    "If a question is outside session scope, it automatically uses general AI."
)

question = st.text_input("Ask a question:")

if st.button("Ask"):
    if question:
        with st.spinner("Agent is reasoning..."):
            if is_session_meta_question(question):
                knowledge = retrieve_relevant_knowledge(question)
                answer = answer_from_sessions(
                    "Provide an overall analysis of all sessions.",
                    knowledge
                )
                st.success(answer)

            elif check_scope(question):
                knowledge = retrieve_relevant_knowledge(question)
                answer = answer_from_sessions(question, knowledge)
                st.success(answer)

            else:
                answer = general_answer(question)
                st.info(answer)
    else:
        st.warning("Please enter a question.")
