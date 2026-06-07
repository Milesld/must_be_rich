#!/usr/bin/env python3
"""数据库初始化 — 执行 ClickHouse 和 MySQL 的建表 DDL。"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CLICKHOUSE_DDL = """
-- 行情数据库
CREATE TABLE IF NOT EXISTS market_daily (
    trade_date  Date,
    code        String,
    name        String,
    open        Float64,
    high        Float64,
    low         Float64,
    close       Float64,
    pre_close   Float64,
    volume      Float64,
    amount      Float64,
    adj_factor  Float64,
    turnover    Float64,
    is_st       UInt8,
    is_suspended UInt8,
    status      LowCardinality(String)
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(trade_date)
ORDER BY (code, trade_date);

CREATE TABLE IF NOT EXISTS market_minute (
    trade_date  Date,
    time        DateTime,
    code        String,
    open        Float64,
    high        Float64,
    low         Float64,
    close       Float64,
    volume      Float64,
    amount      Float64
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(trade_date)
ORDER BY (code, time);

CREATE TABLE IF NOT EXISTS financial_statements (
    code            String,
    report_date     Date,
    announce_date   Date,
    statement_type  String,
    revenue         Float64,
    revenue_yoy     Float64,
    net_profit      Float64,
    net_profit_yoy  Float64,
    gross_margin    Float64,
    net_margin      Float64,
    total_assets    Float64,
    total_liabilities Float64,
    total_equity    Float64,
    debt_ratio      Float64,
    current_ratio   Float64,
    quick_ratio     Float64,
    operating_cf    Float64,
    free_cf         Float64,
    cf_ratio        Float64,
    eps             Float64,
    bps             Float64,
    roe             Float64,
    roa             Float64,
    roic            Float64,
    data_source     String,
    version         String
) ENGINE = ReplacingMergeTree(version)
PARTITION BY toYYYYMM(report_date)
ORDER BY (code, report_date, statement_type);

CREATE TABLE IF NOT EXISTS factor_values (
    factor_name     String,
    code            String,
    calc_date       Date,
    value           Float64,
    version         String
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(calc_date)
ORDER BY (factor_name, code, calc_date);
"""

MYSQL_DDL = """
CREATE TABLE IF NOT EXISTS account (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    name            VARCHAR(64) NOT NULL,
    type            ENUM('core','trading') NOT NULL,
    initial_capital DECIMAL(18,4) NOT NULL,
    current_cash    DECIMAL(18,4) NOT NULL,
    frozen_cash     DECIMAL(18,4) DEFAULT 0,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at      DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS position (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    account_id      INT NOT NULL,
    code            VARCHAR(6) NOT NULL,
    shares          INT NOT NULL,
    avg_cost        DECIMAL(10,4) NOT NULL,
    current_price   DECIMAL(10,4),
    unrealized_pnl  DECIMAL(18,4),
    realized_pnl    DECIMAL(18,4) DEFAULT 0,
    open_date       DATE NOT NULL,
    UNIQUE KEY uk_account_code (account_id, code)
);

CREATE TABLE IF NOT EXISTS signal_log (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    generated_at    DATETIME(3) NOT NULL,
    code            VARCHAR(6) NOT NULL,
    signal_type     VARCHAR(32) NOT NULL,
    direction       ENUM('long','short','flat') NOT NULL,
    strength        DECIMAL(5,4),
    confidence      DECIMAL(5,4),
    model_name      VARCHAR(64) NOT NULL,
    model_version   VARCHAR(32) NOT NULL,
    feature_snapshot JSON NOT NULL,
    expiry_at       DATETIME(3),
    status          ENUM('pending','executed','expired','rejected_by_risk','cancelled'),
    INDEX idx_code_time (code, generated_at)
);

CREATE TABLE IF NOT EXISTS order_log (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    signal_id       BIGINT,
    account_id      INT NOT NULL,
    code            VARCHAR(6) NOT NULL,
    side            ENUM('buy','sell') NOT NULL,
    price_type      ENUM('limit','market') NOT NULL,
    order_price     DECIMAL(10,4),
    order_shares    INT NOT NULL,
    submitted_at    DATETIME(3),
    filled_shares   INT DEFAULT 0,
    filled_avg_price DECIMAL(10,4),
    commission      DECIMAL(10,4) DEFAULT 0,
    stamp_duty      DECIMAL(10,4) DEFAULT 0,
    transfer_fee    DECIMAL(10,4) DEFAULT 0,
    status          ENUM('pending','partial_filled','filled','cancelled','rejected')
);

CREATE TABLE IF NOT EXISTS risk_log (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    timestamp       DATETIME(3) NOT NULL,
    check_type      VARCHAR(64) NOT NULL,
    code            VARCHAR(6),
    account_id      INT,
    order_id        BIGINT,
    rejected_reason VARCHAR(512) NOT NULL,
    context_json    JSON,
    INDEX idx_time (timestamp)
);
"""


def main():
    import clickhouse_driver
    import pymysql

    ch_url = os.environ.get("CLICKHOUSE_URL", "http://localhost:8123")
    mysql_url_raw = os.environ.get("MYSQL_URL", "mysql://quant:quant@localhost:3306/quant")

    # ClickHouse
    raw = ch_url.replace("http://", "").replace("https://", "")
    host, port_str = (raw.split(":") + ["9000"])[:2]
    ch = clickhouse_driver.Client(host=host, port=int(port_str))
    for stmt in CLICKHOUSE_DDL.strip().split(";"):
        stmt = stmt.strip()
        if stmt and not stmt.startswith("--"):
            try:
                ch.execute(stmt)
                print(f"CH OK: {stmt.split()[2]}")
            except Exception as e:
                print(f"CH SKIP: {e}")

    # MySQL
    from urllib.parse import urlparse
    pu = urlparse(mysql_url_raw)
    conn = pymysql.connect(
        host=pu.hostname, port=pu.port or 3306,
        user=pu.username, password=pu.password,
        database=pu.path.lstrip("/"),
    )
    with conn.cursor() as cur:
        for stmt in MYSQL_DDL.strip().split(";"):
            stmt = stmt.strip()
            if stmt and not stmt.startswith("--"):
                try:
                    cur.execute(stmt)
                    print(f"MySQL OK: {stmt.split()[2]}")
                except Exception as e:
                    print(f"MySQL SKIP: {e}")
    conn.commit()
    conn.close()
    print("数据库初始化完成")


if __name__ == "__main__":
    main()
