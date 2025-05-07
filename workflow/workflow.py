from langgraph.graph import StateGraph, END
from typing import Dict, List, Tuple, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument
from langchain_core.messages import HumanMessage, AIMessage
from .agents.base_agent import AgentState, Document
from .tools.fetch_data import get_data_with_indicators
from .tools.ingest_documents import process_and_ingest_documents, ingest_stock_data
from .tools.plot_graph import plot_stock_data, plot_comparison

# Initialize ChromaDB and other components
def get_vectorstore():
    """Initialize and return ChromaDB vectorstore."""
    # Use OpenAI embeddings (can be replaced with other embedding models)
    embeddings = OpenAIEmbeddings()
    
    # Initialize ChromaDB
    vectorstore = Chroma(
        collection_name="stock_analysis_docs",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )
    
    return vectorstore

# Node functions for the workflow
def ingest_documents(state: AgentState) -> AgentState:
    """Process and ingest documents into ChromaDB."""
    try:
        # Convert our Document objects to LangChain Document objects
        langchain_docs = [
            LangchainDocument(
                page_content=doc.page_content,
                metadata=doc.metadata
            ) for doc in state.documents
        ]
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        split_docs = text_splitter.split_documents(langchain_docs)
        
        # Get vectorstore and add documents
        vectorstore = get_vectorstore()
        vectorstore.add_documents(split_docs)
        
        return state
    except Exception as e:
        state.error = f"Error ingesting documents: {str(e)}"
        return state

def retrieve_context(state: AgentState) -> AgentState:
    """Retrieve relevant documents based on the query."""
    try:
        if not state.query:
            state.error = "No query provided for retrieval"
            return state
            
        # Get vectorstore and retrieve documents
        vectorstore = get_vectorstore()
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5} # Returns 5 most similar documents to the query
        )
        
        # Retrieve documents
        docs = retriever.invoke(state.query)
        
        # Convert LangChain documents to our Document objects
        retrieved_docs = [
            Document(
                page_content=doc.page_content,
                metadata=doc.metadata
            ) for doc in docs
        ]
        
        state.retrieved_documents = retrieved_docs
        
        # Create context string from retrieved documents
        context_parts = [f"Document {i+1}:\n{doc.page_content}\n" 
                        for i, doc in enumerate(retrieved_docs)]
        state.context = "\n".join(context_parts)
        
        return state
    except Exception as e:
        state.error = f"Error retrieving context: {str(e)}"
        return state

def generate_response(state: AgentState) -> AgentState:
    """Generate response using LLM with retrieved context."""
    try:
        # Create prompt template
        template = """
        You are a world-class AI stock trading assistant. Your job is to evaluate market data enriched with technical indicators and provide clear, accurate trading recommendations. Follow these instructions carefully:

        1. Rules:
        - You are an expert in technical analysis with extensive knowledge of all major indicators.
        - Always analyze and explain your thought process step-by-step.
        - Use beginner-friendly language but maintain precision.
        - Use only the data provided — do not hallucinate.
        - Provide a recommendation: Buy, Hold, or Sell.
        - Back up your recommendation using the technical indicator values.
        - Cite known technical principles (e.g., "RSI above 70 indicates overbought conditions").

        2. Analyze the following technical data for {ticker}:

        3. Structure your response with these tags:
        <Thought> [Your reasoning and interpretation of the indicators] </Thought>
        <FinalAnswer>
        [Your recommendation: Buy, Hold, or Sell]
        [Justification based on each indicator]
        [References to technical analysis principles]
        </FinalAnswer>

        Current conversation:
        {chat_history}
        
        Context from knowledge base:
        {context}
        
        Technical Indicators:
        {technical_indicators}
        
        User Query: {query}
        
        Provide a detailed analysis and trading recommendation. Include:
        1. Whether to Buy, Hold, or Sell
        2. Suggested entry price
        3. Stop loss level
        4. Target price
        5. Confidence level in your recommendation (low, medium, high)
        
        Base your recommendation on both the technical indicators and the retrieved context.
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Initialize LLM
        llm = ChatOpenAI(model="gpt-4", temperature=0.2)
        
        # Create chain
        chain = prompt | llm | StrOutputParser()
        
        # Get stock data with indicators if not already provided
        if state.ticker and not state.technical_indicators:
            try:
                # Default to last 1 year of data if not specified
                from datetime import datetime, timedelta
                end_date = datetime.now().strftime("%Y-%m-%d")
                start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
                
                # Get data with indicators
                data = get_data_with_indicators(ticker=state.ticker, start_date=start_date, end_date=end_date)
                
                # Extract latest indicators
                latest = data.iloc[-1]
                state.technical_indicators = {
                    "EMA_200": float(latest.get("EMA_200", 0)),
                    "RSI": float(latest.get("RSI", 0)),
                    "MACD": float(latest.get("MACD", 0)),
                    "MACD_Signal": float(latest.get("MACD_Signal", 0)),
                    "BB_High": float(latest.get("BB_High", 0)),
                    "BB_Low": float(latest.get("BB_Low", 0)),
                    "Close": float(latest.get("Close", 0))
                }
                
                # Generate and save plot
                import os
                plots_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots")
                os.makedirs(plots_dir, exist_ok=True)
                plot_path = os.path.join(plots_dir, f"{state.ticker}_analysis.png")
                
                # Create plot
                plot_stock_data(data, state.ticker, save_path=plot_path, show_plot=False)
                
                # Add plot path to state
                state.metadata = state.metadata or {}
                state.metadata["plot_path"] = plot_path
                
            except Exception as e:
                print(f"Warning: Could not generate technical indicators: {str(e)}")
        
        # Prepare inputs
        tech_indicators_str = "\n".join([f"- {k}: {v}" for k, v in state.technical_indicators.items()])
        
        # Format chat history for prompt
        formatted_chat_history = format_chat_history(state.chat_history)
        
        # Run chain
        response = chain.invoke({
            "ticker": state.ticker or "Unknown",
            "context": state.context,
            "technical_indicators": tech_indicators_str,
            "query": state.query,
            "chat_history": formatted_chat_history
        })
        
        state.response = response
        
        # Parse response to extract structured fields
        import re
        
        # Extract recommendation (Buy, Hold, Sell)
        rec_match = re.search(r"(Buy|Hold|Sell)", response)
        if rec_match:
            state.trade_recommendation = rec_match.group(1)
        
        # Extract prices
        entry_match = re.search(r"entry price[:\s]*(\$?[\d\.]+)", response, re.IGNORECASE)
        if entry_match:
            try:
                state.entry_price = float(entry_match.group(1).replace('$', ''))
            except ValueError:
                pass
                
        stop_match = re.search(r"stop loss[:\s]*(\$?[\d\.]+)", response, re.IGNORECASE)
        if stop_match:
            try:
                state.stop_loss = float(stop_match.group(1).replace('$', ''))
            except ValueError:
                pass
                
        target_match = re.search(r"target price[:\s]*(\$?[\d\.]+)", response, re.IGNORECASE)
        if target_match:
            try:
                state.target_price = float(target_match.group(1).replace('$', ''))
            except ValueError:
                pass
        
        # Extract confidence
        conf_match = re.search(r"confidence[:\s]*(low|medium|high)", response, re.IGNORECASE)
        if conf_match:
            confidence_text = conf_match.group(1).lower()
            # Convert text to numeric score
            confidence_map = {"low": 0.3, "medium": 0.6, "high": 0.9}
            state.confidence_score = confidence_map.get(confidence_text, 0.5)
        
        return state
    except Exception as e:
        state.error = f"Error generating response: {str(e)}"
        return state

def handle_error(state: AgentState) -> Dict[str, Any]:
    """Handle errors in the workflow."""
    # Log the error
    print(f"Error in workflow: {state.error}")
    
    # Return END to terminate the workflow
    return {"next": END}

def route_next_step(state: AgentState) -> Dict[str, Any]:
    """Determine the next step in the workflow based on state."""
    if state.error:
        return {"next": "handle_error"}
    
    if not state.retrieved_documents:
        return {"next": "retrieve_context"}
    
    if not state.response:
        return {"next": "generate_response"}
    
    # Workflow complete
    return {"next": END}

def manage_chat_memory(state: AgentState) -> AgentState:
    """Manage conversation history in the agent state."""
    try:
        # If we have a new query and a response, add them to chat history
        if state.query and state.response:
            # Add user message to history
            state.chat_history.append({"role": "user", "content": state.query})
            
            # Add assistant response to history
            state.chat_history.append({"role": "assistant", "content": state.response})
            
            # Limit chat history length to prevent token overflow
            # Keep the most recent 10 messages (5 exchanges)
            if len(state.chat_history) > 10:
                state.chat_history = state.chat_history[-10:]
                
        return state
    except Exception as e:
        state.error = f"Error managing chat memory: {str(e)}"
        return state

def format_chat_history(chat_history: List[Dict[str, str]]) -> str:
    """Format chat history for inclusion in prompts."""
    if not chat_history:
        return ""
        
    formatted_history = []
    for message in chat_history:
        role = message["role"]
        content = message["content"]
        formatted_history.append(f"{role.capitalize()}: {content}")
    
    return "\n".join(formatted_history)

def build_workflow_graph():
    """Build and return the workflow graph."""
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("ingest_documents", ingest_documents)
    workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("generate_response", generate_response)
    workflow.add_node("manage_chat_memory", manage_chat_memory)
    workflow.add_node("handle_error", handle_error)
    workflow.add_node("route", route_next_step)

    # Add Edges
    # Start with routing
    workflow.set_entry_point("route")
    
    # Connect nodes
    workflow.add_edge("route", "retrieve_context")
    workflow.add_edge("route", "generate_response")
    workflow.add_edge("route", "handle_error")
    
    workflow.add_edge("ingest_documents", "route")
    workflow.add_edge("retrieve_context", "route")
    workflow.add_edge("generate_response", "manage_chat_memory")
    workflow.add_edge("manage_chat_memory", "route")

    return workflow

# Create a runnable workflow
def get_runnable_workflow():
    """Create and return a runnable workflow."""
    workflow_graph = build_workflow_graph()
    return workflow_graph.compile()