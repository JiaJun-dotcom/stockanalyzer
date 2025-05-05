import os
import pandas as pd
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, CSVLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from workflow.base_agent import Document

# Load environment variables
load_dotenv()

def load_documents(file_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Load documents from various file formats (txt, csv, pdf)
    
    Args:
        file_paths: List of file paths to load
        
    Returns:
        List of loaded documents
    """
    documents = []
    
    for file_path in file_paths:
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.txt':
                loader = TextLoader(file_path)
                documents.extend(loader.load())
            elif file_extension == '.csv':
                loader = CSVLoader(file_path)
                documents.extend(loader.load())
            elif file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            else:
                print(f"Unsupported file format: {file_extension}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return documents

def process_and_ingest_documents(file_paths: List[str], collection_name: str = "stock_analysis_docs"):
    """
    Process and ingest documents into ChromaDB
    
    Args:
        file_paths: List of file paths to process and ingest
        collection_name: Name of the ChromaDB collection
    """
    # Load documents
    raw_documents = load_documents(file_paths)
    
    if not raw_documents:
        print("No documents loaded. Check file paths and formats.")
        return
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    split_documents = text_splitter.split_documents(raw_documents)
    
    print(f"Loaded {len(raw_documents)} documents and split into {len(split_documents)} chunks")
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings()
    
    # Initialize and populate ChromaDB
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )
    
    # Add documents to vectorstore
    vectorstore.add_documents(split_documents)
    
    print(f"Successfully ingested documents into ChromaDB collection '{collection_name}'")

def ingest_stock_data(ticker: str, start_date: str, end_date: str, collection_name: str = "stock_data"):
    """
    Ingest stock data from yfinance into ChromaDB
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date for data retrieval (YYYY-MM-DD)
        end_date: End date for data retrieval (YYYY-MM-DD)
        collection_name: Name of the ChromaDB collection
    """
    from workflow.tools.fetch_data import get_data_with_indicators
    
    # Get stock data with indicators
    data = get_data_with_indicators(ticker=ticker, start_date=start_date, end_date=end_date)
    
    # Flatten the columns
    data.columns = data.columns.get_level_values(0)
    
    # Convert to documents
    documents = []
    
    # Process by month to create meaningful chunks
    data['Month'] = data.index.to_period('M')
    monthly_groups = data.groupby('Month')
    
    for month, group in monthly_groups:
        # Create a summary for this month
        month_str = str(month)
        start_price = group['Close'].iloc[0]
        end_price = group['Close'].iloc[-1]
        high = group['High'].max()
        low = group['Low'].min()
        avg_volume = group['Volume'].mean()
        
        # Calculate monthly performance
        performance = ((end_price - start_price) / start_price) * 100
        
        # Create document content
        content = f"""{ticker} Monthly Summary for {month_str}\n
        Opening Price: ${start_price:.2f}\n
        Closing Price: ${end_price:.2f}\n
        Monthly High: ${high:.2f}\n
        Monthly Low: ${low:.2f}\n
        Average Daily Volume: {avg_volume:.0f}\n
        Monthly Performance: {performance:.2f}%\n
        Technical Indicators (End of Month):\n
        RSI: {group['RSI'].iloc[-1]:.2f}\n
        MACD: {group['MACD'].iloc[-1]:.2f}\n
        MACD Signal: {group['MACD_Signal'].iloc[-1]:.2f}\n
        EMA 200: {group['EMA_200'].iloc[-1]:.2f}\n
        Bollinger High: {group['BB_High'].iloc[-1]:.2f}\n
        Bollinger Low: {group['BB_Low'].iloc[-1]:.2f}\n
        """
        
        # Create document with metadata
        doc = Document(
            page_content=content,
            metadata={
                "ticker": ticker,
                "date": month_str,
                "type": "monthly_summary",
                "performance": performance
            }
        )
        
        documents.append(doc)
    
    # Convert to LangChain documents
    from langchain_core.documents import Document as LangchainDocument
    langchain_docs = [
        LangchainDocument(
            page_content=doc.page_content,
            metadata=doc.metadata
        ) for doc in documents
    ]
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings()
    
    # Initialize and populate ChromaDB
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )
    
    # Add documents to vectorstore
    vectorstore.add_documents(langchain_docs)
    
    print(f"Successfully ingested {ticker} stock data into ChromaDB collection '{collection_name}'")

if __name__ == "__main__":
    # Example usage
    # 1. Ingest documents from files
    # file_paths = [
    #     "path/to/stock_reports/google_q1_2023.pdf",
    #     "path/to/stock_reports/market_analysis.txt",
    #     "path/to/stock_reports/financial_data.csv"
    # ]
    # process_and_ingest_documents(file_paths)
    
    # 2. Ingest stock data
    # ingest_stock_data("GOOG", "2020-01-01", "2023-04-23")
    
    print("Run this script with your specific file paths and stock tickers.")
    print("Example:")
    print("  process_and_ingest_documents(['path/to/file1.pdf', 'path/to/file2.txt'])")
    print("  ingest_stock_data('AAPL', '2020-01-01', '2023-04-23')")