from enum import Enum

class Chain(Enum):
    ETHEREUM = "ethereum"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"
    POLYGON = "polygon"

class Protocol(Enum):
    UNISWAP = "uniswap"
    AAVE = "aave"
    COMPOUND = "compound"
    MAKERDAO = "makerdao"

class MetricType(Enum):
    TVL = "tvl"
    VOLUME = "volume"
    ACTIVE_USERS = "active_users"
