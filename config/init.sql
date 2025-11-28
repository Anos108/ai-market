-- PostgreSQL initialization script for AI Market Analysis System
-- This script sets up the database with proper configuration

-- Create database if not exists (handled by POSTGRES_DB environment variable)
-- CREATE DATABASE ai_market_system;

-- Connect to the database
\c ai_market_system;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Set timezone
SET timezone = 'UTC';

-- Create custom types
DO $$ BEGIN
    CREATE TYPE symbol_status AS ENUM ('active', 'monitoring', 'inactive', 'watchlist');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE symbol_source AS ENUM ('manual', 'ticker_discovery', 'portfolio', 'recommendation');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE trading_action AS ENUM ('buy', 'sell', 'hold', 'watch');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Create symbols table
CREATE TABLE IF NOT EXISTS symbols (
    symbol VARCHAR(20) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    sector VARCHAR(100) NOT NULL,
    industry VARCHAR(100) NOT NULL,
    market_cap BIGINT,
    price DECIMAL(10,2),
    volume BIGINT,
    description TEXT,
    exchange VARCHAR(20) DEFAULT 'NASDAQ',
    currency VARCHAR(3) DEFAULT 'USD',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create managed_symbols table
CREATE TABLE IF NOT EXISTS managed_symbols (
    symbol VARCHAR(20) PRIMARY KEY,
    status symbol_status NOT NULL DEFAULT 'active',
    source symbol_source NOT NULL DEFAULT 'manual',
    added_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    notes TEXT,
    target_price DECIMAL(10,2),
    stop_loss DECIMAL(10,2),
    position_size DECIMAL(15,2),
    priority INTEGER DEFAULT 1 CHECK (priority BETWEEN 1 AND 5),
    tags JSONB DEFAULT '[]'::jsonb,
    alerts_enabled BOOLEAN DEFAULT TRUE,
    auto_trade_enabled BOOLEAN DEFAULT FALSE,
    initial_price DECIMAL(10,2),
    initial_price_date TIMESTAMP,
    CONSTRAINT fk_managed_symbols_symbol 
        FOREIGN KEY (symbol) REFERENCES symbols (symbol) ON DELETE CASCADE
);

-- Create symbol_performance table for historical data
CREATE TABLE IF NOT EXISTS symbol_performance (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    price DECIMAL(10,2) NOT NULL,
    volume BIGINT,
    change_percent DECIMAL(8,4),
    rsi DECIMAL(5,2),
    sma_20 DECIMAL(10,2),
    volatility DECIMAL(8,4),
    market_cap BIGINT,
    pe_ratio DECIMAL(8,2),
    CONSTRAINT fk_symbol_performance_symbol 
        FOREIGN KEY (symbol) REFERENCES symbols (symbol) ON DELETE CASCADE
);

-- Create trading_decisions table
CREATE TABLE IF NOT EXISTS trading_decisions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    action trading_action NOT NULL,
    confidence DECIMAL(3,2) NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    reasoning TEXT NOT NULL,
    target_price DECIMAL(10,2),
    stop_loss DECIMAL(10,2),
    position_size DECIMAL(15,2),
    timeframe VARCHAR(20) DEFAULT 'short_term',
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_trading_decisions_symbol 
        FOREIGN KEY (symbol) REFERENCES symbols (symbol) ON DELETE CASCADE
);

-- Create agent_signals table for tracking agent predictions
CREATE TABLE IF NOT EXISTS agent_signals (
    id SERIAL PRIMARY KEY,
    agent_name VARCHAR(100) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    signal_type VARCHAR(20) NOT NULL,
    confidence DECIMAL(3,2) NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    reasoning TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb,
    CONSTRAINT fk_agent_signals_symbol 
        FOREIGN KEY (symbol) REFERENCES symbols (symbol) ON DELETE CASCADE
);

-- Create agent_performance table for tracking agent metrics
CREATE TABLE IF NOT EXISTS agent_performance (
    id SERIAL PRIMARY KEY,
    agent_name VARCHAR(100) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_predictions INTEGER DEFAULT 0,
    correct_predictions INTEGER DEFAULT 0,
    accuracy DECIMAL(5,4) DEFAULT 0.0,
    avg_confidence DECIMAL(3,2) DEFAULT 0.0,
    sharpe_ratio DECIMAL(8,4) DEFAULT 0.0,
    win_rate DECIMAL(5,4) DEFAULT 0.0,
    health_score DECIMAL(5,2) DEFAULT 0.0,
    performance_trend VARCHAR(20) DEFAULT 'stable',
    last_prediction_time TIMESTAMP,
    UNIQUE(agent_name, timestamp)
);

-- Create agent_feedback table for tracking prediction outcomes
CREATE TABLE IF NOT EXISTS agent_feedback (
    id SERIAL PRIMARY KEY,
    agent_name VARCHAR(100) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    predicted_signal VARCHAR(20) NOT NULL,
    actual_outcome VARCHAR(20) NOT NULL,
    feedback_score DECIMAL(5,4) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb,
    CONSTRAINT fk_agent_feedback_symbol 
        FOREIGN KEY (symbol) REFERENCES symbols (symbol) ON DELETE CASCADE
);

-- Create online_learning_status table for tracking model training
CREATE TABLE IF NOT EXISTS online_learning_status (
    id SERIAL PRIMARY KEY,
    agent_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(100) NOT NULL,
    model_accuracy DECIMAL(5,4) DEFAULT 0.0,
    training_samples INTEGER DEFAULT 0,
    is_training BOOLEAN DEFAULT FALSE,
    last_training TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    learning_rate DECIMAL(8,6) DEFAULT 0.001,
    epochs_completed INTEGER DEFAULT 0,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(agent_name, timestamp)
);

-- Create agent_routing_decisions table for tracking intelligent routing decisions
CREATE TABLE IF NOT EXISTS agent_routing_decisions (
    id SERIAL PRIMARY KEY,
    decision_id VARCHAR(100) UNIQUE NOT NULL,
    market_regime VARCHAR(50) NOT NULL,
    regime_confidence DECIMAL(5,4) NOT NULL,
    volatility_level DECIMAL(5,4) NOT NULL,
    routing_strategy VARCHAR(100) NOT NULL,
    active_agents TEXT[] NOT NULL,
    decision_confidence DECIMAL(5,4) NOT NULL,
    risk_level VARCHAR(20) NOT NULL,
    expected_performance DECIMAL(5,4),
    actual_performance DECIMAL(5,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create market_regime_detection table for tracking market regime analysis
CREATE TABLE IF NOT EXISTS market_regime_detection (
    id SERIAL PRIMARY KEY,
    regime_type VARCHAR(50) NOT NULL,
    confidence DECIMAL(5,4) NOT NULL,
    volatility_level DECIMAL(5,4) NOT NULL,
    trend_strength DECIMAL(5,4) NOT NULL,
    market_sentiment VARCHAR(50) NOT NULL,
    regime_duration INTEGER NOT NULL,
    transition_probability DECIMAL(5,4) NOT NULL,
    rsi_oversold BOOLEAN NOT NULL,
    macd_bullish BOOLEAN NOT NULL,
    volume_increasing BOOLEAN NOT NULL,
    vix_elevated BOOLEAN NOT NULL,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create agent_routing_weights table for tracking agent weighting decisions
CREATE TABLE IF NOT EXISTS agent_routing_weights (
    id SERIAL PRIMARY KEY,
    agent_name VARCHAR(100) NOT NULL,
    weight DECIMAL(5,4) NOT NULL,
    performance_score DECIMAL(5,4) NOT NULL,
    regime_fitness DECIMAL(5,4) NOT NULL,
    confidence_adjustment DECIMAL(5,4) NOT NULL,
    selection_reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create ticker_discovery_history table
CREATE TABLE IF NOT EXISTS ticker_discovery_history (
    id SERIAL PRIMARY KEY,
    scan_id UUID NOT NULL DEFAULT uuid_generate_v4(),
    scan_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    total_scanned INTEGER NOT NULL DEFAULT 0,
    triggers_found INTEGER NOT NULL DEFAULT 0,
    high_priority INTEGER NOT NULL DEFAULT 0,
    avg_score DECIMAL(5,2) DEFAULT 0.0,
    avg_confidence DECIMAL(5,2) DEFAULT 0.0,
    scan_duration_ms INTEGER DEFAULT 0,
    status VARCHAR(20) DEFAULT 'completed',
    notes TEXT
);

-- Create ticker_discovery_results table for individual discovery results
CREATE TABLE IF NOT EXISTS ticker_discovery_results (
    id SERIAL PRIMARY KEY,
    scan_id UUID NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    trigger_type VARCHAR(50) NOT NULL,
    priority VARCHAR(20) NOT NULL CHECK (priority IN ('LOW', 'MEDIUM', 'HIGH')),
    confidence DECIMAL(5,2) NOT NULL CHECK (confidence BETWEEN 0 AND 100),
    score DECIMAL(5,2) NOT NULL DEFAULT 0.0,
    description TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    discovered_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_discovery_results_scan_id 
        FOREIGN KEY (scan_id) REFERENCES ticker_discovery_history(scan_id) ON DELETE CASCADE,
    CONSTRAINT fk_discovery_results_symbol 
        FOREIGN KEY (symbol) REFERENCES symbols (symbol) ON DELETE CASCADE
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_managed_symbols_status ON managed_symbols(status);
CREATE INDEX IF NOT EXISTS idx_managed_symbols_source ON managed_symbols(source);
CREATE INDEX IF NOT EXISTS idx_managed_symbols_priority ON managed_symbols(priority);
CREATE INDEX IF NOT EXISTS idx_managed_symbols_added_date ON managed_symbols(added_date);
CREATE INDEX IF NOT EXISTS idx_managed_symbols_last_updated ON managed_symbols(last_updated);

CREATE INDEX IF NOT EXISTS idx_symbol_performance_symbol ON symbol_performance(symbol);
CREATE INDEX IF NOT EXISTS idx_symbol_performance_timestamp ON symbol_performance(timestamp);
CREATE INDEX IF NOT EXISTS idx_symbol_performance_symbol_timestamp ON symbol_performance(symbol, timestamp);

CREATE INDEX IF NOT EXISTS idx_trading_decisions_symbol ON trading_decisions(symbol);
CREATE INDEX IF NOT EXISTS idx_trading_decisions_generated_at ON trading_decisions(generated_at);
CREATE INDEX IF NOT EXISTS idx_trading_decisions_action ON trading_decisions(action);
CREATE INDEX IF NOT EXISTS idx_trading_decisions_confidence ON trading_decisions(confidence);

CREATE INDEX IF NOT EXISTS idx_discovery_history_timestamp ON ticker_discovery_history(scan_timestamp);
CREATE INDEX IF NOT EXISTS idx_discovery_history_status ON ticker_discovery_history(status);
CREATE INDEX IF NOT EXISTS idx_discovery_results_symbol ON ticker_discovery_results(symbol);
CREATE INDEX IF NOT EXISTS idx_discovery_results_priority ON ticker_discovery_results(priority);
CREATE INDEX IF NOT EXISTS idx_discovery_results_scan_id ON ticker_discovery_results(scan_id);
CREATE INDEX IF NOT EXISTS idx_discovery_results_discovered_at ON ticker_discovery_results(discovered_at);

-- Create GIN index for JSONB tags
CREATE INDEX IF NOT EXISTS idx_managed_symbols_tags ON managed_symbols USING GIN (tags);

-- Create partial indexes for common queries
CREATE INDEX IF NOT EXISTS idx_managed_symbols_active ON managed_symbols(symbol) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_symbol_performance_recent ON symbol_performance(symbol, timestamp) WHERE timestamp > NOW() - INTERVAL '7 days';
CREATE INDEX IF NOT EXISTS idx_trading_decisions_recent ON trading_decisions(symbol, generated_at) WHERE generated_at > NOW() - INTERVAL '24 hours';

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
DROP TRIGGER IF EXISTS update_symbols_updated_at ON symbols;
CREATE TRIGGER update_symbols_updated_at
    BEFORE UPDATE ON symbols
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_managed_symbols_updated_at ON managed_symbols;
CREATE TRIGGER update_managed_symbols_updated_at
    BEFORE UPDATE ON managed_symbols
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Create a view for symbol summary
CREATE OR REPLACE VIEW symbol_summary AS
SELECT 
    s.symbol,
    s.name,
    s.sector,
    s.industry,
    ms.status,
    ms.source,
    ms.priority,
    ms.target_price,
    ms.stop_loss,
    ms.added_date,
    ms.last_updated,
    sp.price as current_price,
    sp.change_percent,
    sp.rsi,
    sp.volatility,
    td.action as latest_decision,
    td.confidence as decision_confidence,
    td.generated_at as decision_time
FROM symbols s
LEFT JOIN managed_symbols ms ON s.symbol = ms.symbol
LEFT JOIN LATERAL (
    SELECT price, change_percent, rsi, volatility
    FROM symbol_performance
    WHERE symbol = s.symbol
    ORDER BY timestamp DESC
    LIMIT 1
) sp ON true
LEFT JOIN LATERAL (
    SELECT action, confidence, generated_at
    FROM trading_decisions
    WHERE symbol = s.symbol
    ORDER BY generated_at DESC
    LIMIT 1
) td ON true;

-- Create a function to get symbol performance history
CREATE OR REPLACE FUNCTION get_symbol_performance_history(
    p_symbol VARCHAR(20),
    p_days INTEGER DEFAULT 30
)
RETURNS TABLE (
    timestamp TIMESTAMP,
    price DECIMAL(10,2),
    volume BIGINT,
    change_percent DECIMAL(8,4),
    rsi DECIMAL(5,2),
    volatility DECIMAL(8,4)
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        sp.timestamp,
        sp.price,
        sp.volume,
        sp.change_percent,
        sp.rsi,
        sp.volatility
    FROM symbol_performance sp
    WHERE sp.symbol = p_symbol
      AND sp.timestamp > NOW() - INTERVAL '1 day' * p_days
    ORDER BY sp.timestamp DESC;
END;
$$ LANGUAGE plpgsql;

-- Create a function to clean old performance data
CREATE OR REPLACE FUNCTION clean_old_performance_data(p_days INTEGER DEFAULT 90)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM symbol_performance 
    WHERE timestamp < NOW() - INTERVAL '1 day' * p_days;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Insert some sample data
INSERT INTO symbols (symbol, name, sector, industry) VALUES
    ('AAPL', 'Apple Inc.', 'Technology', 'Consumer Electronics'),
    ('MSFT', 'Microsoft Corporation', 'Technology', 'Software'),
    ('GOOGL', 'Alphabet Inc.', 'Technology', 'Internet Content & Information'),
    ('TSLA', 'Tesla Inc.', 'Consumer Discretionary', 'Auto Manufacturers'),
    ('AMZN', 'Amazon.com Inc.', 'Consumer Discretionary', 'Internet Retail')
ON CONFLICT (symbol) DO NOTHING;

-- Grant permissions (if using a different user)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO ai_market_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO ai_market_user;
-- Create execution_orders table for tracking order management
CREATE TABLE IF NOT EXISTS execution_orders (
    id SERIAL PRIMARY KEY,
    order_id VARCHAR(100) UNIQUE NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    order_type VARCHAR(20) NOT NULL CHECK (order_type IN ('market', 'limit', 'stop', 'stop_limit')),
    side VARCHAR(10) NOT NULL CHECK (side IN ('buy', 'sell')),
    quantity DECIMAL(15,4) NOT NULL CHECK (quantity > 0),
    price DECIMAL(15,4),
    stop_price DECIMAL(15,4),
    status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'submitted', 'filled', 'partially_filled', 'cancelled', 'rejected', 'expired')),
    strategy VARCHAR(100) NOT NULL,
    commission DECIMAL(10,4) DEFAULT 0.0,
    execution_time_seconds INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    filled_at TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_execution_orders_symbol 
        FOREIGN KEY (symbol) REFERENCES symbols (symbol) ON DELETE CASCADE
);

-- Create execution_positions table for tracking current positions
CREATE TABLE IF NOT EXISTS execution_positions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    quantity DECIMAL(15,4) NOT NULL,
    average_price DECIMAL(15,4) NOT NULL,
    market_value DECIMAL(15,2) NOT NULL,
    unrealized_pnl DECIMAL(15,2) NOT NULL DEFAULT 0.0,
    realized_pnl DECIMAL(15,2) NOT NULL DEFAULT 0.0,
    total_commission DECIMAL(10,4) NOT NULL DEFAULT 0.0,
    position_type VARCHAR(20) NOT NULL DEFAULT 'long' CHECK (position_type IN ('long', 'short')),
    strategy VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_execution_positions_symbol 
        FOREIGN KEY (symbol) REFERENCES symbols (symbol) ON DELETE CASCADE
);

-- Create execution_strategies table for tracking execution strategies
CREATE TABLE IF NOT EXISTS execution_strategies (
    id SERIAL PRIMARY KEY,
    strategy_name VARCHAR(100) UNIQUE NOT NULL,
    strategy_type VARCHAR(50) NOT NULL CHECK (strategy_type IN ('twap', 'vwap', 'iceberg', 'adaptive', 'aggressive', 'passive')),
    parameters JSONB NOT NULL DEFAULT '{}',
    performance_metrics JSONB NOT NULL DEFAULT '{}',
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    total_orders INTEGER NOT NULL DEFAULT 0,
    successful_orders INTEGER NOT NULL DEFAULT 0,
    avg_execution_time DECIMAL(8,2) NOT NULL DEFAULT 0.0,
    success_rate DECIMAL(5,4) NOT NULL DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for execution tables
CREATE INDEX IF NOT EXISTS idx_execution_orders_symbol ON execution_orders(symbol);
CREATE INDEX IF NOT EXISTS idx_execution_orders_status ON execution_orders(status);
CREATE INDEX IF NOT EXISTS idx_execution_orders_created_at ON execution_orders(created_at);
CREATE INDEX IF NOT EXISTS idx_execution_positions_symbol ON execution_positions(symbol);
CREATE INDEX IF NOT EXISTS idx_execution_strategies_active ON execution_strategies(is_active);

-- Create rag_news_documents table for storing news documents
CREATE TABLE IF NOT EXISTS rag_news_documents (
    id SERIAL PRIMARY KEY,
    doc_id VARCHAR(100) UNIQUE NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    source VARCHAR(100) NOT NULL,
    url TEXT NOT NULL,
    category VARCHAR(50) NOT NULL,
    tags TEXT[] DEFAULT '{}',
    embedding VECTOR(384),
    similarity_score DECIMAL(5,4),
    published_at TIMESTAMP NOT NULL,
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create rag_analysis table for storing RAG analysis results
CREATE TABLE IF NOT EXISTS rag_analysis (
    id SERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    llm_response TEXT NOT NULL,
    confidence DECIMAL(5,4) NOT NULL,
    reasoning TEXT NOT NULL,
    analysis_type VARCHAR(50) NOT NULL DEFAULT 'market_impact',
    relevant_doc_ids TEXT[] DEFAULT '{}',
    response_time_ms INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create rag_performance table for tracking RAG system performance
CREATE TABLE IF NOT EXISTS rag_performance (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(10,4) NOT NULL,
    metric_unit VARCHAR(20),
    measurement_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT
);

-- Create indexes for RAG tables
CREATE INDEX IF NOT EXISTS idx_rag_news_documents_source ON rag_news_documents(source);
CREATE INDEX IF NOT EXISTS idx_rag_news_documents_published_at ON rag_news_documents(published_at);
CREATE INDEX IF NOT EXISTS idx_rag_news_documents_category ON rag_news_documents(category);
CREATE INDEX IF NOT EXISTS idx_rag_analysis_created_at ON rag_analysis(created_at);
CREATE INDEX IF NOT EXISTS idx_rag_performance_metric_name ON rag_performance(metric_name);
CREATE INDEX IF NOT EXISTS idx_rag_performance_measurement_date ON rag_performance(measurement_date);

-- Create rl_training_metrics table for storing RL training data
CREATE TABLE IF NOT EXISTS rl_training_metrics (
    id SERIAL PRIMARY KEY,
    algorithm VARCHAR(50) NOT NULL,
    episodes_trained INTEGER NOT NULL DEFAULT 0,
    avg_episode_reward DECIMAL(10,4) NOT NULL DEFAULT 0.0,
    best_episode_reward DECIMAL(10,4) NOT NULL DEFAULT 0.0,
    convergence_episode INTEGER,
    training_loss DECIMAL(10,6) NOT NULL DEFAULT 0.0,
    exploration_rate DECIMAL(5,4) NOT NULL DEFAULT 0.1,
    experience_buffer_size INTEGER NOT NULL DEFAULT 0,
    model_accuracy DECIMAL(5,4) NOT NULL DEFAULT 0.0,
    training_duration_seconds INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create rl_performance_metrics table for storing RL performance data
CREATE TABLE IF NOT EXISTS rl_performance_metrics (
    id SERIAL PRIMARY KEY,
    total_return DECIMAL(10,4) NOT NULL DEFAULT 0.0,
    sharpe_ratio DECIMAL(8,4) NOT NULL DEFAULT 0.0,
    sortino_ratio DECIMAL(8,4) NOT NULL DEFAULT 0.0,
    calmar_ratio DECIMAL(8,4) NOT NULL DEFAULT 0.0,
    max_drawdown DECIMAL(8,4) NOT NULL DEFAULT 0.0,
    win_rate DECIMAL(5,4) NOT NULL DEFAULT 0.0,
    avg_trade_pnl DECIMAL(10,4) NOT NULL DEFAULT 0.0,
    volatility DECIMAL(8,4) NOT NULL DEFAULT 0.0,
    total_trades INTEGER NOT NULL DEFAULT 0,
    profitable_trades INTEGER NOT NULL DEFAULT 0,
    measurement_period VARCHAR(20) NOT NULL DEFAULT '30d',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create rl_actions table for storing RL agent actions
CREATE TABLE IF NOT EXISTS rl_actions (
    id SERIAL PRIMARY KEY,
    action_type VARCHAR(20) NOT NULL CHECK (action_type IN ('buy', 'sell', 'hold', 'strong_buy', 'strong_sell')),
    symbol VARCHAR(20) NOT NULL,
    confidence DECIMAL(5,4) NOT NULL,
    expected_return DECIMAL(8,4) NOT NULL,
    risk_score DECIMAL(5,4) NOT NULL,
    state_features JSONB NOT NULL DEFAULT '{}',
    reward DECIMAL(10,4),
    action_reasoning TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_rl_actions_symbol 
        FOREIGN KEY (symbol) REFERENCES symbols (symbol) ON DELETE CASCADE
);

-- Create indexes for RL tables
CREATE INDEX IF NOT EXISTS idx_rl_training_metrics_algorithm ON rl_training_metrics(algorithm);
CREATE INDEX IF NOT EXISTS idx_rl_training_metrics_created_at ON rl_training_metrics(created_at);
CREATE INDEX IF NOT EXISTS idx_rl_performance_metrics_created_at ON rl_performance_metrics(created_at);
CREATE INDEX IF NOT EXISTS idx_rl_performance_metrics_measurement_period ON rl_performance_metrics(measurement_period);
CREATE INDEX IF NOT EXISTS idx_rl_actions_symbol ON rl_actions(symbol);
CREATE INDEX IF NOT EXISTS idx_rl_actions_action_type ON rl_actions(action_type);
CREATE INDEX IF NOT EXISTS idx_rl_actions_created_at ON rl_actions(created_at);

-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO ai_market_user;

-- Log completion
DO $$
BEGIN
    RAISE NOTICE 'AI Market Analysis System database initialized successfully!';
    RAISE NOTICE 'Database: %', current_database();
    RAISE NOTICE 'User: %', current_user;
    RAISE NOTICE 'Timezone: %', current_setting('timezone');
END $$;