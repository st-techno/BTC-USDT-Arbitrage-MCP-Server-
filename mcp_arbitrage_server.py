import os
import logging
import asyncio
from fastmcp import FastMCP
import ccxt
import numpy as np
from redis import Redis
from redisvl.schema import FieldSchema, IndexSchema
from redisvl.index import SearchIndex
from datetime import datetime
from cerbos_sdk.client import CerbosClient
from cerbos_sdk.check import CheckResourcesRequest, Resource, Principal

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    handlers=[logging.FileHandler("mcp_arbitrage_prod.log"), logging.StreamHandler()]
)
logger = logging.getLogger("MCP-Arbitrage-Server")

# --- MCP Server Instance ---
mcp = FastMCP(name="Institutional BTC-USDT Perps Arbitrage MCP Server with Risk Management")
logger.info("MCP server instance created.")

# --- Secure API Keys & Redis Config ---
EXCHANGE_KEYS = {
    'binance': {
        'apiKey': os.getenv("BINANCE_API_KEY", ""),
        'secret': os.getenv("BINANCE_API_SECRET", "")
    },
    'bitmex': {
        'apiKey': os.getenv("BITMEX_API_KEY", ""),
        'secret': os.getenv("BITMEX_API_SECRET", "")
    },
    'okx': {
        'apiKey': os.getenv("OKX_API_KEY", ""),
        'secret': os.getenv("OKX_API_SECRET", "")
    }
}

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# --- Exchanges Initialization ---
exchanges = {}
for name, creds in EXCHANGE_KEYS.items():
    try:
        exchanges[name] = getattr(ccxt, name)({
            'apiKey': creds['apiKey'],
            'secret': creds['secret'],
            'enableRateLimit': True,
            'options': {'adjustForTimeDifference': True},
        })
        logger.info(f"{name} exchange initialized")
    except Exception as e:
        logger.error(f"Failed to initialize {name}: {e}")
        raise

# --- Redis Vector Store Setup ---
redis_client = Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=False,
    health_check_interval=30,
    socket_keepalive=True,
    socket_timeout=5,
)

schema = IndexSchema.from_dict({
    "index": {
        "name": "arb_agent_memory_idx",
        "prefix": ["state_memory:"],
        "storage_type": "hash",
    },
    "fields": [
        FieldSchema("timestamp", "text"),
        FieldSchema("prices_vector", "vector", attrs={
            "algorithm": "hnsw",
            "datatype": "float32",
            "dims": 3,
            "distance_metric": "cosine",
            "initial_cap": 100_000,
            "m": 16,
            "efConstruction": 200,
            "efRuntime": 100,
        })
    ]
})

search_index = SearchIndex(schema, redis_client)
try:
    search_index.create()
    logger.info("Created Redis vector search index.")
except Exception as e:
    logger.warning(f"Redis index creation error: {e}")

# --- Memory Store with Redis Vector ---
class MemoryStoreRedis:
    def __init__(self, index: SearchIndex):
        self.index = index
        self.redis = index._redis

    def save_state(self, state_vec: np.ndarray):
        try:
            vec_bytes = state_vec.astype(np.float32).tobytes()
            timestamp = datetime.utcnow().isoformat()
            key = f"state_memory:{timestamp}"
            self.redis.hset(key, mapping={
                "timestamp": timestamp.encode(),
                "prices_vector": vec_bytes
            })
            logger.info(f"Saved state vector at {timestamp}")
        except Exception as e:
            logger.error(f"MemoryStore save_state error: {e}")

    def recall_states(self, query_vec: np.ndarray, top_k: int = 5):
        try:
            vec_bytes = query_vec.astype(np.float32).tobytes()
            results = self.index.search(
                vector=vec_bytes,
                k=top_k,
                return_fields=["timestamp", "prices_vector"]
            )
            recalled = []
            for doc in results.docs:
                vec = np.frombuffer(doc["prices_vector"], dtype=np.float32)
                ts = doc["timestamp"].decode()
                recalled.append({"timestamp": ts, "vector": vec.tolist()})
            return recalled
        except Exception as e:
            logger.error(f"MemoryStore recall_states error: {e}")
            return []

memory_store = MemoryStoreRedis(search_index)
logger.info("Redis memory store initialized.")

# --- Cerbos Authorization Client Setup ---
CERBOS_HOST = os.getenv("CERBOS_HOST", "http://localhost:3592")
cerbos_client = CerbosClient(base_url=CERBOS_HOST)

async def authorize(principal_id: str, tool_id: str, context: dict) -> bool:
    request = CheckResourcesRequest(
        principal=Principal(id=principal_id),
        resource=Resource(kind="mcp_tool", id=tool_id),
        actions=["invoke"],
        context=context,
    )
    results = await cerbos_client.check_resources(request)
    return results[0].is_allowed()

# --- MCP Tools ---
@mcp.tool()
async def fetch_prices(principal_id: str) -> dict:
    if not await authorize(principal_id, "fetch_prices", {}):
        raise PermissionError("Authorization failed for fetch_prices")
    prices = {}
    for name, ex in exchanges.items():
        try:
            ticker = await asyncio.to_thread(ex.fetch_ticker, "BTC/USDT")
            prices[name] = ticker['last']
        except Exception as e:
            logger.error(f"Price fetch error for {name}: {e}")
            prices[name] = None
    logger.info(f"Fetched prices: {prices}")
    return prices

@mcp.tool()
def save_state(principal_id: str, prices: list) -> str:
    if not asyncio.run(authorize(principal_id, "save_state", {})):
        raise PermissionError("Authorization failed for save_state")
    arr = np.array(prices, dtype=np.float32)
    memory_store.save_state(arr)
    logger.info(f"Memory state saved by principal: {principal_id}")
    return "State saved"

@mcp.tool()
def recall_states(principal_id: str, query: list, top_k: int = 5) -> list:
    if not asyncio.run(authorize(principal_id, "recall_states", {})):
        raise PermissionError("Authorization failed for recall_states")
    arr = np.array(query, dtype=np.float32)
    recalled = memory_store.recall_states(arr, top_k)
    logger.info(f"States recalled by principal: {principal_id}")
    return recalled

@mcp.tool()
def plan_arbitrage(principal_id: str, prices: dict, threshold: float = 25.0,
                   max_drawdown: float = 0.05, max_position: float = 1.0) -> dict:
    if not asyncio.run(authorize(principal_id, "plan_arbitrage", {})):
        raise PermissionError("Authorization failed for plan_arbitrage")
    valid_prices = {k: v for k, v in prices.items() if v is not None}
    if len(valid_prices) < 2:
        return {}
    min_ex = min(valid_prices, key=valid_prices.get)
    max_ex = max(valid_prices, key=valid_prices.get)
    spread = valid_prices[max_ex] - valid_prices[min_ex]
    current_drawdown = 0.03  # Ideally, fetched dynamically from state/risk engine
    if spread > threshold and current_drawdown <= max_drawdown:
        plan = {"buy_exchange": min_ex, "sell_exchange": max_ex, "amount": 0.02}
        logger.info(f"Arbitrage plan for {principal_id}: {plan}")
        return plan
    else:
        logger.info(f"No arbitrage for {principal_id}. Spread: {spread}, Drawdown: {current_drawdown}")
        return {}

@mcp.tool()
def risk_check(principal_id: str, drawdown: float, position: float,
               max_drawdown: float = 0.05, max_position: float = 1.0) -> bool:
    if not asyncio.run(authorize(principal_id, "risk_check", {})):
        raise PermissionError("Authorization failed for risk_check")
    if drawdown > max_drawdown:
        logger.warning(f"Drawdown breach by {principal_id}")
        return False
    if abs(position) > max_position:
        logger.warning(f"Position size breach by {principal_id}")
        return False
    return True

@mcp.tool()
def backtest(principal_id: str, historical_data: list, initial_balance: float = 10000.0) -> dict:
    if not asyncio.run(authorize(principal_id, "backtest", {})):
        raise PermissionError("Authorization failed for backtest")
    balance = initial_balance
    pnl_history = []
    position = 0
    drawdown = 0
    for prices in historical_data:
        plan = plan_arbitrage(principal_id, prices)
        if plan and risk_check(principal_id, drawdown, position):
            buy_price = prices.get(plan["buy_exchange"], 0)
            sell_price = prices.get(plan["sell_exchange"], 0)
            amount = plan["amount"]
            realized_pnl = (sell_price - buy_price) * amount
            balance += realized_pnl
            position += amount
            drawdown = min(drawdown, balance / initial_balance - 1)
            memory_store.save_state(np.array(list(prices.values()), dtype=np.float32))
        pnl_history.append(balance)
        logger.debug(f"Backtest step for {principal_id}: pnl={realized_pnl}, balance={balance}, position={position}, drawdown={drawdown}")
    returns = np.diff(pnl_history) / initial_balance if len(pnl_history) > 1 else np.array([0])
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
    sortino = (np.mean(returns) / np.sqrt(np.mean(np.square(returns[returns < 0]))) if (returns < 0).any() else 0)
    results = {"final_balance": balance, "sharpe_ratio": sharpe, "sortino_ratio": sortino}
    logger.info(f"Backtest for {principal_id}: {results}")
    return results

@mcp.tool()
async def place_order(principal_id: str, exchange_name: str, symbol: str, side: str, amount: float) -> dict:
    if not await authorize(principal_id, "place_order", {}):
        raise PermissionError("Authorization failed for place_order")
    logger.info(f"Mock order: {side} {amount} {symbol} on {exchange_name} by {principal_id}")
    # For real implementation, switch to orders via exchange API (asyncio.to_thread for blocking calls)
    return {"status": "success", "message": f"Order placed: {side} {amount} {symbol} on {exchange_name}"}

# --- MCP Server Entrypoint ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting MCP server...")
    uvicorn.run(mcp.get_asgi_app(), host="0.0.0.0", port=8080)
