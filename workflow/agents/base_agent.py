from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

class Document(BaseModel):
    """Document class for storing text and metadata."""
    page_content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AgentState(BaseModel):
    """State for the RAG workflow agent."""
    # Input fields
    query: str = ""
    technical_indicators: Dict[str, float] = Field(default_factory=dict)
    ticker: Optional[str] = None
    
    # RAG processing fields
    documents: List[Document] = Field(default_factory=list)
    retrieved_documents: List[Document] = Field(default_factory=list)
    context: str = ""
    
    # Chat history
    chat_history: List[Dict[str, str]] = Field(default_factory=list)

    # Agent scratchpad
    agent_scratchpad: str = ""

    # Output fields
    response: str = ""
    trade_recommendation: Optional[str] = None
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    target_price: Optional[float] = None
    confidence_score: Optional[float] = None
    
    # Additional metadata (for storing plot paths, etc.)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Error handling
    error: Optional[str] = None
