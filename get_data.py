from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import threading
import time
import csv
from datetime import datetime

# Define a class to handle the IB API connection and data callbacks
class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.data = []  # List to store historical data
        self.data_received = threading.Event()  # Event to signal when data is received
        self.symbol = None # Store the symbol for filename

    def historicalData(self, reqId, bar):
        """Callback function to handle incoming historical data"""
        date_str_from_ib = bar.date
        try:
            # Split the date string by space
            parts = date_str_from_ib.split()
            date_time_str = " ".join(parts[:2]) # Take first two parts: date and time
            timezone_str = parts[2] if len(parts) > 2 else None # Get timezone if available

            # Parse the date and time string (without timezone for now)
            date_obj = datetime.strptime(date_time_str, "%Y%m%d %H:%M:%S")

            formatted_date_str = date_obj.strftime("%Y-%m-%d %H:%M:%S") # Format to YYYY-MM-DD HH:MM:SS
            self.data.append({
                'date': formatted_date_str, # Use the formatted date string
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume
            })
        except ValueError as e:
            print(f"Error parsing date string: {date_str_from_ib}. Error: {e}")
            print(f"Parts of the date string: {parts}") # Debugging: print the parts
            date_obj = None
            print(f"Skipping bar due to date parsing error: Date: {bar.date}, Open: {bar.open}, High: {bar.high}, Low: {bar.low}, Close: {bar.close}, Volume: {bar.volume}, WAP: {bar.wap}, BarCount: {bar.barCount}")


    def historicalDataEnd(self, reqId, start, end):
        """Callback when all historical data has been received"""
        print(f"Historical data received for reqId {reqId}")
        if self.symbol:
            file_path = f'{self.symbol}_5min_historical_data.csv' # More descriptive filename
        else:
            file_path = 'historical_data.csv' # default if symbol is not set
        self.write_to_csv(file_path)
        self.data_received.set()

    def write_to_csv(self, file_path):
        """Write the historical data to a CSV file"""
        if not self.data:
            print("No data to write to CSV.")
            return
        with open(file_path, 'w', newline='') as csvfile:
            fieldnames = ['date', 'open', 'high', 'low', 'close', 'volume']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.data:
                writer.writerow(row)
        print(f"Data successfully written to {file_path}")

# Function to create a stock contract
def create_stock_contract(symbol):
    """Create a contract for the specified stock symbol"""
    contract = Contract()
    contract.symbol = symbol
    contract.secType = "STK"  # Stock
    contract.exchange = "SMART"  # Smart routing
    contract.currency = "USD"  # US Dollar
    return contract

# Function to run the API message loop
def run_loop(app):
    app.run()

# Main function to execute the program
def main():
    # Initialize the IB API client
    app = IBapi()
    app.connect("127.0.0.1", 7497, 123)  # Adjust host, port, clientId as needed

    # Start the API message loop in a separate thread
    api_thread = threading.Thread(target=run_loop, args=(app,), daemon=True)
    api_thread.start()

    # Wait briefly for the connection to establish
    time.sleep(1)

    symbols = ['QQQ', 'SPY'] # Get data for all symbols

    for ticker_symbol in symbols:
        print(f"Fetching historical data for {ticker_symbol}...")
        # Store the ticker symbol in the IBapi instance
        app.symbol = ticker_symbol

        # Clear previous data
        app.data = []
        app.data_received.clear()

        # Create a contract for the desired stock
        contract = create_stock_contract(ticker_symbol)

        # Request historical data
        app.reqHistoricalData(
            reqId=1,                  # Request ID (can be the same for sequential requests)
            contract=contract,        # Stock contract
            endDateTime="",           # Empty string for current time
            durationStr="3 Y",        # 200 days of data
            barSizeSetting="5 mins",  # 5-minute bars
            whatToShow="TRADES",      # Show trade data
            useRTH=1,                 # Regular trading hours only
            formatDate=1,             # Format date as string (important for parsing)
            keepUpToDate=False,       # No real-time updates
            chartOptions=[]           # No additional options
        )

        # Wait for the data to be received (timeout after 300 seconds)
        app.data_received.wait(timeout=300)
        if not app.data_received.is_set():
            print(f"Timeout waiting for data for {ticker_symbol}. No CSV file was written.")
        else:
            print(f"Historical data CSV file created for {ticker_symbol}.")
        time.sleep(2) # Small delay between requests


    # Disconnect from the API after all symbols are processed
    app.disconnect()
    print("Data fetching and CSV creation completed for all symbols.")

# Entry point of the program
if __name__ == "__main__":
    main()