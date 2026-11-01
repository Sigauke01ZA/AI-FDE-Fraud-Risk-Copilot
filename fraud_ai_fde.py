import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_classic.chains import RetrievalQA

# --- 1. PATHS & SETUP ---
BASE_DIR = Path(__file__).resolve().parent.parent # Points to Project_5_Fraud_Detection
MODEL_PATH = BASE_DIR / "models" / "fraud_xgb_model.pkl"
FEATURES_PATH = BASE_DIR / "models" / "fraud_features.pkl"
POLICY_PATH = BASE_DIR / "ai_fde" / "Fraud_Security_Policy.txt"

# Load Model
model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)

# --- 2. SETUP RAG (Ollama) ---
@st.cache_resource
def setup_rag():
    loader = TextLoader(str(POLICY_PATH))
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    
    llm = ChatOllama(model="llama3", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa_chain

st.set_page_config(page_title="Fraud AI FDE Copilot", layout="wide")
st.title("🛡️ Fraud Detection AI Copilot")
st.markdown("Simulate a transaction, see the model's decision, and then ask the AI *why* the decision was made based on the Fraud Security Policy.")

# --- 3. SIDEBAR: TRANSACTION INPUTS ---
st.sidebar.header("Transaction Profile")
amount = st.sidebar.number_input("Transaction Amount (R)", 1.0, 50000.0, 500.0)
deviation = st.sidebar.slider("Avg Amount Deviation (Sigma)", 0.0, 10.0, 1.0)
geo_velocity = st.sidebar.slider("Geo-Velocity (km/h)", 0.0, 2000.0, 10.0)
device_match = st.sidebar.selectbox("Device Fingerprint Match", ["Known Device", "Unknown Device", "New Device", "Spoofed"])
velocity_normal = st.sidebar.selectbox("Transaction Velocity (1h)", ["Within Normal", "Burst Pattern", "5x Baseline"])
cvv_status = st.sidebar.selectbox("CVV Match Status", ["Match", "Mismatch", "Not Provided"])

# --- 4. PREDICTION LOGIC ---
# Create a numeric DataFrame with all zeros
input_df = pd.DataFrame(0.0, index=[0], columns=features)

# 🚨 THE FIX: Inject realistic defaults for the "hidden" features
input_df['session_duration_sec'] = 120.0       # Assume 2-minute checkout
input_df['account_age_days'] = 365.0           # Assume 1-year-old account
input_df['failed_attempts_before_success'] = 0.0
input_df['chargeback_history_count'] = 0.0

# Now apply the user's slider inputs
input_df['transaction_amount'] = amount
input_df['avg_amount_deviation_sigma'] = deviation
input_df['geo_velocity_kmh'] = geo_velocity

if device_match == "Unknown Device": input_df['device_fingerprint_match_unknown_device'] = 1.0
elif device_match == "New Device": input_df['device_fingerprint_match_new_device'] = 1.0
elif device_match == "Spoofed": input_df['device_fingerprint_match_spoofed'] = 1.0

if velocity_normal == "Burst Pattern": input_df['transaction_velocity_1h_burst_pattern'] = 1.0
elif velocity_normal == "5x Baseline": input_df['transaction_velocity_1h_5x_baseline'] = 1.0

if cvv_status == "Mismatch": input_df['cvv_match_status_cvv2_mismatch'] = 1.0
elif cvv_status == "Not Provided": input_df['cvv_match_status_not_provided'] = 1.0

fraud_prob = model.predict_proba(input_df)[0][1]

# --- 5. DISPLAY RESULTS ---
st.header("Model Decision")
col1, col2 = st.columns(2)
col1.metric("Fraud Probability", f"{fraud_prob:.2%}")

if fraud_prob > 0.50:
    decision = "BLOCK TRANSACTION"
    col2.metric("Action", decision)
    st.error(f"🚨 **ACTION: {decision}.** This transaction exhibits strong patterns of fraud.")
elif fraud_prob > 0.20:
    decision = "MANUAL REVIEW"
    col2.metric("Action", decision)
    st.warning(f"⚠️ **ACTION: {decision}.** Send an OTP or SMS verification to the customer.")
else:
    decision = "APPROVE"
    col2.metric("Action", decision)
    st.success(f"✅ **ACTION: {decision}.** Transaction appears legitimate.")

# --- 6. AI CHAT INTERFACE ---
st.divider()
st.subheader("🤖 Ask the AI Copilot")
st.markdown("*Example: 'Why was this transaction blocked?' or 'What does the policy say about impossible travel?'*")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about this transaction or the fraud policy..."):
    st.chat_message("user").markdown(prompt)
    
    # Inject context into the prompt
    context_prompt = f"""
    You are an expert AI Fraud Copilot for a bank.
    Here is the data for the simulated transaction:
    - Amount: R{amount}
    - Amount Deviation (Sigma): {deviation}
    - Geo-Velocity: {geo_velocity} km/h
    - Device Match: {device_match}
    - Transaction Velocity: {velocity_normal}
    - CVV Status: {cvv_status}
    - Model Fraud Probability: {fraud_prob:.2%}
    - Model Action: {decision}
    
    User Question: {prompt}
    
    Answer the question accurately using the transaction data and the Fraud Security Policy. Cite the policy section if applicable.
    """
    
    with st.spinner("Analyzing fraud policy and transaction data..."):
        qa_chain = setup_rag()
        response = qa_chain.run(context_prompt)
    
    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": response})