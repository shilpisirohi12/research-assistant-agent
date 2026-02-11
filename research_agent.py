import os
from dotenv import load_dotenv
# from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

# Load .env file
load_dotenv()


# LLM setup (Groq with a strong model for reasoning)
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5)  # Lower temp for more focused research

# Custom tool: Save report to file
def save_to_file(content: str, filename: str = "research_report.md") -> str:
    """Saves the given content to a Markdown file and returns the saved path."""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Report saved successfully to {filename}"
    except Exception as e:
        return f"Error saving file: {str(e)}"

# Convert it to a LangChain tool
save_tool = StructuredTool.from_function(
    func=save_to_file,
    name="save_to_file",
    description="Saves the final research report to a Markdown file on disk. Use this ONLY when you have the complete final report ready. Input: the full Markdown content and optional filename.",
    args_schema=None,  # No strict schema needed for simple string inputs
)

# Search tool 
search_tool = TavilySearch(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
)
tools = [search_tool,save_tool]  # List of tools (expandable later)

# Bind tools to LLM so it can call them
llm_with_tools = llm.bind_tools(tools)


# system_prompt = """
# You are a helpful research assistant that ALWAYS uses tools when you need external information.

# For any user query:
# - If you need up-to-date, specific, or detailed info, you MUST call the tavily_search_results_json tool (possibly multiple times with different queries).
# - Do NOT guess or make up information — search instead.
# - After gathering enough data, output ONLY the final Markdown report.
# - Structure the report like this:
#   ## Final Research Report: [Query]
#   ### Overview
#   [1-2 sentence summary]
#   ### Top Activities / Facts
#   1. [Item] - [brief description] [source link if available]
#   2. ...
#   ### Sources
#   - [url1]
#   - [url2]
# - Aim for comprehensive but concise answers (e.g., for "100 things" try to list as many as reasonably possible from search results).
# """


system_prompt = """
You are a helpful research assistant that ALWAYS uses tools when you need external information.

For any user query:
- If you need up-to-date, specific, or detailed info, you MUST call the tavily_search_results_json tool (possibly multiple times with different queries).
- Do NOT guess or make up information — search instead.
- After gathering enough data, analyze and decide if more searches are needed.
- When you have enough information and the report is complete, output a final Markdown report.
- Structure the report like this:
  ## Final Research Report: [Query]
  ### Overview
  [1-2 sentence summary]
  ### Top Activities / Facts
  1. [Item] - [brief description] [source link if available]
  2. ...
  ### Sources
  - [url1]
  - [url2]
- Aim for comprehensive but concise answers (e.g., for "100 things" try to list as many as reasonably possible from search results).
- After writing the final report, ALWAYS call the save_to_file tool with the full report content as input. This is mandatory for completing the task.
- Only output the final report when ready. Do not include intermediate thoughts in the final response.

Tool calling format: Use tools exactly when needed.

Current date is around February 2026 — prioritize recent info if relevant.
"""
# Agent node: LLM decides next action
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),                    # Static system instructions
        MessagesPlaceholder(variable_name="messages"),  # Dynamic history goes here
    ]
)
# agent = prompt | llm_with_tools | StrOutputParser()
agent = prompt | llm_with_tools 

# Updated agent_node to handle raw AIMessage with tool_calls
def agent_node(state):
    # Invoke with dict input
    response = agent.invoke({"messages": state["messages"]})
    
    # If it's a tool call, return as-is
    # If it's text, wrap as AIMessage
    if isinstance(response, str):
        response = AIMessage(content=response)
    
    return {"messages": [response]}

# Tool node: Runs the tool if called
tool_node = ToolNode(tools=tools)

# Graph setup
graph = StateGraph(state_schema=MessagesState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_edge("tools", "agent")  # After tool, back to agent
graph.add_conditional_edges("agent", tools_condition)  # Decide: tool or end
graph.set_entry_point("agent")

# Compile the graph into a runnable app
research_agent = graph.compile()

def run_research(query: str, max_steps: int = 15):
    messages = [{"role": "user", "content": query}]
    return research_agent.invoke(
        {"messages": messages},
        config={"recursion_limit": max_steps}
    )

# Quick test: Run the agent on a query
if __name__ == "__main__":
    query = "100 things to do in Toronto"
    
    messages = [{"role": "user", "content": query}]
    print("Starting agent with query:", query)
    print("Initial messages:", messages)
    
    try:
        result = run_research(query=query,max_steps=10)
        print("\n--- Full result keys ---")
        print(result.keys())
        
        print("\n--- All messages in history ---")
        for i, msg in enumerate(result["messages"]):
            print(f"[{i}] {msg}")
            if hasattr(msg, 'content'):
                print(f"   Content: {msg.content[:200]}...")  # truncate long ones
            print("-" * 50)
        
        if result["messages"]:
            final = result["messages"][-1]
            print("\nFinal output:")
            print(final.content if hasattr(final, 'content') else final)
        else:
            print("No messages returned!")
            
    except Exception as e:
        print("Error during invoke:", str(e))
        import traceback
        traceback.print_exc()
