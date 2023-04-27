# -*- coding: utf-8 -*-
"""
author: zengbin93
email: zeng_bin8888@163.com
create_dt: 2023/3/10 23:39
describe: Tushare 数据接口
"""
import os
from czsc import data

dc = data.TsDataCache(data_path=os.environ.get('ts_data_path', r'.\ts_data'))


def get_symbols(step):
    return data.get_symbols(dc, step)


def get_raw_bars(symbol, freq, sdt, edt, fq='后复权', raw_bar=True):
    """读取本地数据"""
    ts_code, asset = symbol.split("#")
    freq = str(freq)
    adj = "qfq" if fq == "前复权" else "hfq"

    if "分钟" in freq:
        freq = freq.replace("分钟", "min")
        bars = dc.pro_bar_minutes(ts_code, sdt=sdt, edt=edt, freq=freq, asset=asset, adj=adj, raw_bar=raw_bar)

    else:
        _map = {"日线": "D", "周线": "W", "月线": "M"}
        freq = _map[freq]
        bars = dc.pro_bar(ts_code, start_date=sdt, end_date=edt, freq=freq, asset=asset, adj=adj, raw_bar=raw_bar)
    return bars
