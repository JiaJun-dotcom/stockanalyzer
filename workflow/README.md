# Stock Analyzer RAG Workflow

This module implements a Retrieval-Augmented Generation (RAG) workflow using ChromaDB for the Stock Analyzer project. The workflow enables context-aware stock analysis by combining technical indicators with relevant information retrieved from a knowledge base. It now includes chat memory to maintain conversation history between interactions.

## Components

### 1. Base Agent State (`base_agent.py`)

Defines the state structure for the RAG workflow:

- Input fields: query, technical indicators, ticker
- RAG processing fields: documents, retrieved documents, context
- Chat memory: maintains conversation history between interactions
- Output fields: response, trade recommendation, entry/exit prices, confidence score

### 2. Workflow Graph (`workflow.py`)

Implements a directed graph workflow using LangGraph with the following nodes:

- `ingest_documents`: Processes and stores documents in ChromaDB
- `retrieve_context`: Retrieves relevant documents based on the query
- `generate_response`: Creates a comprehensive analysis using LLM with retrieved context
- `manage_chat_memory`: Maintains conversation history between interactions
- `handle_error`: Manages error conditions in the workflow
- `route`: Determines the next step based on the current state

### 3. Document Ingestion (`ingest_documents.py`)

Utilities for populating the ChromaDB vector store:

- Load documents from various file formats (TXT, CSV, PDF)
- Process and chunk documents for efficient retrieval
- Ingest historical stock data with technical indicators

### 4. Example Usage (`example_usage.py`)

Demonstrates how to use the RAG workflow:

- Initialize the workflow
- Create documents and ingest them into ChromaDB
- Set up a query with technical indicators
- Execute the workflow and get results

## Getting Started

1. Ensure you have the required dependencies:
   ```
   pip install langchain langchain-openai langchain-core langchain-community langgraph chromadb pydantic
   ```

2. Set up your environment variables in `.env`:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ```

3. Ingest documents into ChromaDB:
   ```python
   from workflow.ingest_documents import process_and_ingest_documents, ingest_stock_data
   
   # Ingest documents from files
   process_and_ingest_documents(['path/to/file1.pdf', 'path/to/file2.txt'])
   
   # Ingest historical stock data
   ingest_stock_data('AAPL', '2020-01-01', '2023-04-23')
   ```

4. Run the workflow with chat memory:
   ```python
   from workflow.example_usage import main
   
   # Initialize workflow
   workflow = main()
   
   # First query
   state = AgentState(query="What's your analysis of AAPL stock?", ticker="AAPL")
   result = workflow.invoke(state)
   print(result.response)
   
   # Follow-up query with chat memory
   follow_up_state = AgentState(
       query="What about its comparison to other tech stocks?",
       ticker="AAPL",
       chat_history=result.chat_history  # Pass the chat history from previous interaction
   )
   follow_up_result = workflow.invoke(follow_up_state)
   print(follow_up_result.response)
   ```

## Customization

- Modify the prompt template in `generate_response()` to customize the analysis
- Adjust retrieval parameters in `retrieve_context()` to control document relevance
- Configure chat memory settings in `manage_chat_memory()` to adjust history retention
- Extend the `AgentState` class to include additional fields for your specific use case

### Chat Memory Configuration

The chat memory feature can be customized in several ways:

- Adjust the maximum number of messages retained in history (default: 10)
- Modify the formatting of chat history in the `format_chat_history()` function
- Implement persistence for chat history by saving/loading from a database
- Add message filtering or preprocessing in the `manage_chat_memory()` function