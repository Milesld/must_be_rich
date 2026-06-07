"""盘前推荐模型组。

包含四个子模型：
- overnight_mapping: 隔夜海外市场→个股开盘方向映射
- gap_classifier: 开盘跳空（高开/平开/低开）三分类
- fusion_ranker: 长期评分+隔夜信号→综合推荐排序
- auction_anomaly: 集合竞价异常检测
"""
