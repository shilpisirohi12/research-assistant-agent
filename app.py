import streamlit as st
from dotenv import load_dotenv
import os
from research_agent import research_agent  # ← import your compiled graph/agent


load_dotenv()  # loads GROQ_API_KEY, TAVILY_API_KEY from .env

# Page config
st.set_page_config(
    page_title="Personal Research Assistant",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Personal Research Assistant")
st.markdown("Ask any question — the agent will search the web, reason, and give you a summarized Markdown report.")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to research?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Show thinking spinner
    with st.chat_message("assistant"):
        with st.spinner("Researching... (this may take 30–90 seconds)"):
            try:
                # Prepare input for your LangGraph agent
                input_messages = [{"role": "user", "content": prompt}]

                # Invoke your agent (with recursion limit for safety)
                result = research_agent.invoke(
                    {"messages": input_messages},
                    config={"recursion_limit": 20}
                )

                # Get the final answer (last message content)
                final_answer = result["messages"][-1].content

                # Display it
                st.markdown(final_answer)

                # Save to session history
                st.session_state.messages.append({"role": "assistant", "content": final_answer})

            except Exception as e:
                st.error(f"Something went wrong: {str(e)}")
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}"})