import logging
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.common import TickerId, BarData
from threading import Thread, Lock
import queue
import time
import pandas as pd
from datetime import datetime, timedelta

class IBKRInterface(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.data = {}
        self.data_queue = queue.Queue()
        self.lock = Lock()
        self.logger = self.setup_logger()
        self.next_valid_order_id = None
        self.contract_details = {}

    def setup_logger(self):
        logger = logging.getLogger('IBKRInterface')
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler('ibkr_interface.log')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger

    def connect(self, host='127.0.0.1', port=7497, clientId=1):
        super().connect(host, port, clientId)
        self.run_thread = Thread(target=self.run)
        self.run_thread.start()
        self.logger.info(f"Connected to IBKR: {host}:{port}")
        while self.next_valid_order_id is None:
            time.sleep(0.1)

    def disconnect(self):
        self.done = True
        super().disconnect()
        self.run_thread.join()
        self.logger.info("Disconnected from IBKR")

    def error(self, reqId: TickerId, errorCode: int, errorString: str):
        self.logger.error(f"Error {errorCode}: {errorString}")

    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)
        self.next_valid_order_id = orderId
        self.logger.info(f"Next valid order ID: {orderId}")

    def historicalData(self, reqId: int, bar: BarData):
        with self.lock:
            if reqId not in self.data:
                self.data[reqId] = []
            self.data[reqId].append(bar)

    def historicalDataEnd(self, reqId: int, start: str, end: str):
        self.data_queue.put(self.data[reqId])
        self.logger.info(f"Historical data received for reqId: {reqId}")

    def get_historical_data(self, contract, duration='5 D', bar_size='1 hour', what_to_show='MIDPOINT'):
        reqId = self.next_valid_order_id
        self.next_valid_order_id += 1
        
        self.reqHistoricalData(reqId, contract, '', duration, bar_size, what_to_show, 1, 1, False, [])
        
        try:
            data = self.data_queue.get(timeout=10)
            df = pd.DataFrame([(bar.date, bar.open, bar.high, bar.low, bar.close, bar.volume) for bar in data],
                              columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%d %H:%M:%S')
            df.set_index('timestamp', inplace=True)
            return df
        except queue.Empty:
            self.logger.warning(f"Timeout waiting for historical data: {contract.symbol}")
            return None

    def place_order(self, contract, order):
        orderId = self.next_valid_order_id
        self.next_valid_order_id += 1
        self.placeOrder(orderId, contract, order)
        self.logger.info(f"Order placed: {order.action} {order.totalQuantity} {contract.symbol}")
        return orderId

    def create_contract(self, symbol, sec_type='CASH', exchange='IDEALPRO', currency='USD'):
        contract = Contract()
        contract.symbol = symbol[:3]
        contract.secType = sec_type
        contract.exchange = exchange
        contract.currency = symbol[3:]
        return contract

    def create_order(self, action, quantity, order_type='MKT'):
        order = Order()
        order.action = action
        order.totalQuantity = quantity
        order.orderType = order_type
        return order

    def get_account_summary(self):
        self.reqAccountSummary(1, "All", "NetLiquidation,TotalCashValue,AvailableFunds")

    def accountSummary(self, reqId: int, account: str, tag: str, value: str, currency: str):
        self.logger.info(f"Account Summary - {account}: {tag} = {value} {currency}")

    def get_contract_details(self, contract):
        self.reqContractDetails(self.next_valid_order_id, contract)
        self.next_valid_order_id += 1

    def contractDetails(self, reqId: int, contractDetails):
        self.contract_details[reqId] = contractDetails

    def contractDetailsEnd(self, reqId: int):
        self.logger.info(f"Contract details received for reqId: {reqId}")

    def get_tick_size(self, contract):
        if contract.symbol not in self.contract_details:
            self.get_contract_details(contract)
            time.sleep(1)  # Wait for contract details to be received
        
        if contract.symbol in self.contract_details:
            return self.contract_details[contract.symbol].minTick
        else:
            self.logger.warning(f"Unable to get tick size for {contract.symbol}")
            return 0.00001  # Default to a small value for forex

    def calculate_position_size(self, account_value, risk_per_trade, stop_loss_percent):
        return int(account_value * risk_per_trade / stop_loss_percent)

def create_ibkr_interface():
    ib = IBKRInterface()
    ib.connect()
    return ib

if __name__ == '__main__':
    ib = create_ibkr_interface()
    
    # Example usage
    contract = ib.create_contract('EURUSD')
    df = ib.get_historical_data(contract, duration='5 D', bar_size='1 hour')
    print(f"Historical data for EURUSD:\n{df.head()}")
    
    ib.get_account_summary()
    
    # Place a market order to buy 10,000 units of EURUSD
    order = ib.create_order('BUY', 10000)
    ib.place_order(contract, order)
    
    time.sleep(5)  # Wait for order to be processed
    ib.disconnect()