import os
import sys
from datetime import datetime, timedelta

# Add the project root to the path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import workflow components
from workflow.workflow import get_runnable_workflow
from workflow.agents.base_agent import AgentState, Document
from workflow.tools.get_data_with_indicators import get_data_with_indicators
from workflow.tools.ingest_documents import process_and_ingest_documents, ingest_stock_data
from workflow.tools.plot_graph import plot_stock_data, plot_comparison

def run_example_workflow():
    """Run an example workflow to demonstrate all components working together."""
    print("\n=== Stock Analyzer Workflow Example ===")
    
    # 1. Set up example data
    ticker = "AAPL"
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    print(f"\nFetching data for {ticker} from {start_date} to {end_date}...")
    
    # 2. Get stock data with indicators
    try:
        data = get_data_with_indicators(ticker=ticker, start_date=start_date, end_date=end_date)
        print(f"Successfully retrieved data with indicators for {ticker}")
        
        # 3. Create and save a plot
        plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
        os.makedirs(plots_dir, exist_ok=True)
        plot_path = os.path.join(plots_dir, f"{ticker}_analysis.png")
        
        plot_result = plot_stock_data(data, ticker, save_path=plot_path, show_plot=False)
        print(f"Created plot and saved to {plot_path}")
        
        # 4. Ingest stock data into ChromaDB
        print("\nIngesting stock data into ChromaDB...")
        ingest_stock_data(ticker, start_date, end_date)
        
        # 5. Set up and run the workflow
        print("\nSetting up workflow...")
        workflow = get_runnable_workflow()
        
        # Create initial state
        state = AgentState(
            query="What is the current trend for Apple stock and should I buy it?",
            ticker=ticker
        )
        
        # Run the workflow
        print("\nRunning workflow...")
        result = workflow.invoke(state)
        
        # Display results
        print("\n=== Workflow Results ===")
        print(f"Recommendation: {result.trade_recommendation}")
        print(f"Entry Price: ${result.entry_price:.2f}" if result.entry_price else "Entry Price: Not specified")
        print(f"Stop Loss: ${result.stop_loss:.2f}" if result.stop_loss else "Stop Loss: Not specified")
        print(f"Target Price: ${result.target_price:.2f}" if result.target_price else "Target Price: Not specified")
        print(f"Confidence: {result.confidence_score:.2f}" if result.confidence_score else "Confidence: Not specified")
        
        print("\nDetailed Analysis:")
        print(result.response)
        
        if "plot_path" in result.metadata:
            print(f"\nPlot saved to: {result.metadata['plot_path']}")
            
    except Exception as e:
        print(f"Error running workflow: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_example_workflow()