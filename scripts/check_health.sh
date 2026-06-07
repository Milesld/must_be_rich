#!/bin/bash
# 系统健康检查 — 遍历所有服务的 /health 端点
# 用法: ./check_health.sh [host]
#   ./check_health.sh          # 默认 localhost
#   ./check_health.sh 10.0.1.5 # 远程主机

HOST="${1:-localhost}"
declare -A ENDPOINTS=(
    [feature_server]="50051/health"
    [risk_engine]="50052/health"
    [nlp_service]="50053/health"
    [monitor]="9090/health"
)

echo "=== 量化系统健康检查 ($HOST) ==="
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

ALL_OK=true

for svc in "${!ENDPOINTS[@]}"; do
    url="http://${HOST}:${ENDPOINTS[$svc]}"
    if response=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 3 "$url" 2>/dev/null); then
        if [ "$response" = "200" ]; then
            echo "  ✓ $svc"
        else
            echo "  ✗ $svc (HTTP $response)"
            ALL_OK=false
        fi
    else
        echo "  ✗ $svc (连接失败)"
        ALL_OK=false
    fi
done

echo ""
if $ALL_OK; then
    echo "✓ 所有服务正常"
    exit 0
else
    echo "✗ 部分服务异常，请检查 docker-compose logs"
    exit 1
fi
