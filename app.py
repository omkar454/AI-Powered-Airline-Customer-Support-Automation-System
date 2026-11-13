import os
import random
import requests
import joblib
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Airline AI Customer Support Desk",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.05rem; color: #4B5563; margin-bottom: 1.5rem; }
    .badge-resolved { background-color: #DCFCE7; color: #166534; padding: 4px 12px; border-radius: 9999px; font-weight: 600; font-size: 0.85rem; }
    .badge-escalated { background-color: #FEE2E2; color: #991B1B; padding: 4px 12px; border-radius: 9999px; font-weight: 600; font-size: 0.85rem; }
    .metric-card { background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 15px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CACHED RESOURCE INITIALIZATION
# ==========================================
@st.cache_resource(show_spinner="Loading AI Models and Vector Database...")
def load_resources():
    # Load environment variables
    load_dotenv("RAG-part/.env")
    
    # 1. Intent Classifier
    intent_model = joblib.load("intent_classifier/intent_classification_hybrid_model.joblib")
    
    # 2. Embedding Model
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    # 3. FAISS Vector Store
    vector_db = FAISS.load_local(
        folder_path="RAG-part/vector-db",
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
    
    # 4. Groq LLM
    llm = ChatGroq(
        model_name="openai/gpt-oss-20b",
        temperature=0.0
    )
    
    return intent_model, embedding_model, vector_db, llm

intent_model, embedding_model, vector_db, llm = load_resources()

# ==========================================
# 3. RAG CHAIN SETUP
# ==========================================
rephrase_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given a chat history and the latest user question, formulate a standalone question. Do NOT answer it, just rewrite if needed."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])
rephrase_chain = rephrase_prompt | llm

email_system_prompt = """You are an official Airline Customer Support Representative for IndiGo.

Write your response as a formal, courteous, and helpful customer support email.
Follow this email structure strictly:
1. Salutation: "Dear Passenger," (or use their name if provided).
2. Opening: Briefly thank them for contacting customer support.
3. Body: Clearly explain the policy details strictly using the provided Context. Use bullet points for limits, fees, or steps.
4. Closing: "If you have any further questions or require additional assistance, please feel free to reply to this email."
5. Sign-off:
   Warm regards,
   IndiGo Customer Support Team
   24/7 Helpline: 0124-6173838 | www.goindigo.in

Strict Grounding Rules:
- Only include facts directly supported by the Context.
- Never invent fees, baggage weights, or refund policies.
- If the Context lacks the answer, politely write:
  "We are currently reviewing your specific request with our ticketing department. A customer support representative will follow up with you shortly."

Context:
{context}
"""

email_prompt_template = ChatPromptTemplate.from_messages([
    ("system", email_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

# ==========================================
# 4. MOCK AIRLINE SERVICE & DATA
# ==========================================
MOCK_FLIGHTS = {
    "6E-204": {"route": "Delhi (DEL) to Mumbai (BOM)", "time": "14:30", "status": "ON TIME", "gate": "14", "terminal": "T2"},
    "6E-551": {"route": "Bengaluru (BLR) to Goa (GOI)", "time": "18:00", "status": "DELAYED (New departure: 18:45)", "gate": "06", "terminal": "T1"},
    "6E-101": {"route": "Mumbai (BOM) to Dubai (DXB)", "time": "21:15", "status": "ON TIME", "gate": "08", "terminal": "T2"}
}

MOCK_BOOKINGS = {
    "IND123": {"name": "Rahul Sharma", "flight": "6E-204", "route": "DEL -> BOM", "seat": "12A", "status": "CONFIRMED"},
    "FLY789": {"name": "Priya Patel", "flight": "6E-551", "route": "BLR -> GOI", "seat": "07C", "status": "CHECKED-IN"}
}

MOCK_PRICES = {
    "DEL-BOM": "₹4,499 (Standard Economy) / ₹6,299 (Flexi Plus)",
    "BOM-GOI": "₹3,150 (Standard Economy) / ₹4,800 (Flexi Plus)",
    "BLR-DEL": "₹5,200 (Standard Economy) / ₹7,100 (Flexi Plus)"
}

TRANSACTIONAL_INTENTS = {
    "check_flight_status", "check_flight_schedule",
    "check_flight_reservation", "check_trip_details",
    "check_flight_prices", "check_trip_prices",
    "book_flight", "book_trip"
}

def mock_airline_service(intent, user_message):
    msg_upper = user_message.upper()
    if "status" in intent or "schedule" in intent:
        for flight_no, details in MOCK_FLIGHTS.items():
            if flight_no in msg_upper or flight_no.replace("-", "") in msg_upper:
                return (
                    f"Dear Passenger,\n\n"
                    f"Thank you for contacting IndiGo regarding flight {flight_no}.\n\n"
                    f"Here are your latest flight details:\n"
                    f"• Flight Number: {flight_no}\n"
                    f"• Route: {details['route']}\n"
                    f"• Scheduled Departure: {details['time']}\n"
                    f"• Current Status: {details['status']}\n"
                    f"• Terminal & Gate: Terminal {details['terminal']}, Gate {details['gate']}\n\n"
                    f"Please arrive at the boarding gate at least 25 minutes prior to departure.\n\n"
                    f"Warm regards,\nIndiGo Customer Support Team\nwww.goindigo.in"
                )
        return (
            "Dear Passenger,\n\n"
            "Thank you for contacting IndiGo flight status service.\n\n"
            "Our scheduled domestic flights are currently operating on schedule. "
            "To view the real-time status of your specific flight, please reply with your "
            "flight number (e.g., 6E-204) or visit www.goindigo.in/flight-status.\n\n"
            "Warm regards,\nIndiGo Customer Support Team\nwww.goindigo.in"
        )
    elif "reservation" in intent or "details" in intent:
        for pnr, b in MOCK_BOOKINGS.items():
            if pnr in msg_upper:
                return (
                    f"Dear {b['name']},\n\n"
                    f"Thank you for contacting IndiGo reservations.\n\n"
                    f"Here are your confirmed reservation details (PNR: {pnr}):\n"
                    f"• Passenger Name: {b['name']}\n"
                    f"• Flight: {b['flight']} ({b['route']})\n"
                    f"• Seat: {b['seat']}\n"
                    f"• Status: {b['status']}\n\n"
                    f"Web check-in opens 48 hours prior to departure.\n\n"
                    f"Warm regards,\nIndiGo Customer Support Team\nwww.goindigo.in"
                )
        return (
            "Dear Passenger,\n\n"
            "We could not locate an active booking with the reference provided. "
            "Please verify your 6-character PNR (e.g., IND123) and reply to this email.\n\n"
            "Warm regards,\nIndiGo Customer Support Team\nwww.goindigo.in"
        )
    elif "prices" in intent:
        return (
            f"Dear Passenger,\n\n"
            f"Thank you for inquiring about IndiGo flight fares.\n\n"
            f"Current starting fares on popular routes:\n"
            f"• Delhi to Mumbai: {MOCK_PRICES['DEL-BOM']}\n"
            f"• Mumbai to Goa: {MOCK_PRICES['BOM-GOI']}\n"
            f"• Bengaluru to Delhi: {MOCK_PRICES['BLR-DEL']}\n\n"
            f"Fares include standard 7 kg cabin baggage and 15 kg checked baggage.\n\n"
            f"Warm regards,\nIndiGo Customer Support Team\nwww.goindigo.in"
        )
    else:
        new_pnr = "6E" + str(random.randint(1000, 9999))
        return (
            f"Dear Passenger,\n\n"
            f"Thank you for initiating your flight booking with IndiGo.\n\n"
            f"Your provisional reservation has been generated:\n"
            f"• Provisional PNR: {new_pnr}\n"
            f"• Booking Status: Awaiting Payment\n\n"
            f"Please complete your reservation within 2 hours at www.goindigo.in/pay.\n\n"
            f"Warm regards,\nIndiGo Customer Support Team\nwww.goindigo.in"
        )

def generate_email_subject(intent, resolved=True):
    clean_topic = intent.replace("check_", "").replace("get_", "").replace("_", " ").title()
    if resolved:
        return f"IndiGo Customer Support: Your Inquiry Regarding {clean_topic}"
    else:
        return f"[ESCALATION REQUIRED] Support Ticket: {clean_topic}"

# ==========================================
# 5. n8n WEBHOOK DISPATCHER
# ==========================================
N8N_PRODUCTION_URL = "https://intern.app.n8n.cloud/webhook/fa48a815-ee95-4229-a17e-5c38f6d28530"

def trigger_n8n_webhook(customer_email, subject, answer, resolved):
    payload = {
        "customer_email": customer_email,
        "subject": subject,
        "answer": answer,
        "resolved": resolved
    }
    try:
        response = requests.post(N8N_PRODUCTION_URL, json=payload, timeout=10)
        return response.status_code == 200, response.status_code
    except Exception as e:
        return False, str(e)

# ==========================================
# 6. ORCHESTRATOR FUNCTION
# ==========================================
def process_ticket(customer_email, user_message, chat_history):
    # Step A: Intent & Confidence
    intent = intent_model.predict([user_message])[0]
    confidence = float(max(intent_model.predict_proba([user_message])[0]))
    
    sources = []
    
    # Step B: Routing
    if confidence < 0.50 or intent == "human_agent":
        resolved = False
        subject = generate_email_subject(intent, resolved=False)
        answer = (
            f"*** INTERNAL ESCALATION TICKET ***\n\n"
            f"Customer Email: {customer_email}\n"
            f"Detected Intent: {intent} (Confidence: {confidence:.1%})\n"
            f"Customer Message: \"{user_message}\"\n\n"
            f"Reason for Escalation:\n"
            f"• Query requires human intervention or confidence fell below 0.50.\n\n"
            f"Action Required:\n"
            f"Please review the customer's request and follow up directly."
        )
        route_type = "Human Escalation Desk"
    elif intent in TRANSACTIONAL_INTENTS:
        resolved = True
        subject = generate_email_subject(intent, resolved=True)
        answer = mock_airline_service(intent, user_message)
        route_type = "Mock Airline API"
    else:
        # Policy RAG
        if not chat_history:
            search_query = user_message
        else:
            search_query = rephrase_chain.invoke({
                "chat_history": chat_history,
                "question": user_message
            }).content
            
        docs = vector_db.similarity_search(search_query, k=4)
        context_text = "\n\n---\n\n".join(d.page_content for d in docs)
        sources = list(set(d.metadata.get("source", "Unknown") for d in docs))
        
        format_prompt = email_prompt_template.invoke({
            "context": context_text,
            "chat_history": chat_history,
            "question": user_message
        })
        response = llm.invoke(format_prompt)
        answer = response.content
        
        if "I do not have specific policy information" in answer or "reviewing your specific request" in answer:
            resolved = False
            subject = generate_email_subject("unresolved_policy_query", resolved=False)
            answer = (
                f"*** INTERNAL ESCALATION TICKET - POLICY GAP ***\n\n"
                f"Customer Email: {customer_email}\n"
                f"Customer Message: \"{user_message}\"\n\n"
                f"Reason for Escalation:\n"
                f"• The knowledge base does not contain the specific policy rules requested.\n\n"
                f"Action Required:\n"
                f"Please assist the passenger directly with verified airline guidelines."
            )
            route_type = "Policy Gap -> Escalation"
        else:
            resolved = True
            subject = generate_email_subject(intent, resolved=True)
            route_type = "RAG Knowledge Engine"
            
    # Step C: Dispatch to n8n
    dispatched, status_code = trigger_n8n_webhook(customer_email, subject, answer, resolved)
    
    # Update History
    chat_history.append(HumanMessage(content=user_message))
    chat_history.append(AIMessage(content=answer))
    
    return {
        "intent": intent,
        "confidence": confidence,
        "route_type": route_type,
        "resolved": resolved,
        "subject": subject,
        "answer": answer,
        "sources": sources,
        "dispatched": dispatched,
        "status_code": status_code
    }

# ==========================================
# 7. STREAMLIT USER INTERFACE
# ==========================================
st.markdown("<div class='main-header'>✈️ Airline Customer Support & Automation Portal</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Powered by ML Intent Classification, RAG Policy Engine, and n8n Gmail Automation</div>", unsafe_allow_html=True)

# Session State for Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "tickets" not in st.session_state:
    st.session_state.tickets = []

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ System Configuration")
    customer_email = st.text_input("Customer Email Address", value="rautom41105@gmail.com")
    
    st.markdown("---")
    st.subheader("💡 Demo Quick Queries")
    if st.button("✈️ Baggage Policy (RAG)"):
        st.session_state.demo_query = "What is the baggage allowance for my flight?"
    if st.button("🔍 Flight Status (Mock API)"):
        st.session_state.demo_query = "Can you please check the status of flight 6E-204?"
    if st.button("🧑‍💼 Speak to Human (Escalation)"):
        st.session_state.demo_query = "I have an urgent issue and need to speak to a human representative right now!"
    if st.button("💰 Refund Policy (RAG)"):
        st.session_state.demo_query = "Can I get a refund for my cancelled flight?"
        
    st.markdown("---")
    if st.button("🔄 Reset Conversation History"):
        st.session_state.chat_history = []
        st.session_state.tickets = []
        st.rerun()

# Main Input Section
default_val = st.session_state.get("demo_query", "")
user_query = st.text_area("Enter Customer Inquiry / Message:", value=default_val, height=90, placeholder="e.g. Can I carry extra baggage on my flight?")

col1, col2 = st.columns([1, 4])
with col1:
    submit = st.button("🚀 Process & Dispatch Ticket", type="primary")

if submit and user_query.strip():
    with st.spinner("Processing through ML Intent Classifier & Orchestrator..."):
        result = process_ticket(customer_email, user_query, st.session_state.chat_history)
        st.session_state.tickets.insert(0, result)

# Display Tickets History
if st.session_state.tickets:
    st.markdown("---")
    st.subheader("📋 Ticket Resolution & Automation Feed")
    
    for idx, t in enumerate(st.session_state.tickets):
        with st.container():
            st.markdown(f"### Ticket #{len(st.session_state.tickets) - idx}: {t['subject']}")
            
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Detected Intent", t['intent'])
            with c2:
                st.metric("Confidence Score", f"{t['confidence']:.2%}")
            with c3:
                st.metric("Routed Engine", t['route_type'])
            with c4:
                status_badge = '<span class="badge-resolved">✅ RESOLVED (Customer Email)</span>' if t['resolved'] else '<span class="badge-escalated">⚠️ ESCALATED (Agent Email)</span>'
                st.markdown(f"**Resolution:**<br>{status_badge}", unsafe_allow_html=True)
            
            st.markdown(f"**n8n Webhook Status:** {'🟢 Dispatched Successfully (HTTP ' + str(t['status_code']) + ')' if t['dispatched'] else '🔴 Failed to Dispatch'}")
            
            if t['sources']:
                st.caption(f"📚 **Policy Documents Cited:** {', '.join(t['sources'])}")
                
            with st.expander("✉️ View Generated Gmail Body", expanded=(idx == 0)):
                st.text(t['answer'])
            st.markdown("---")

