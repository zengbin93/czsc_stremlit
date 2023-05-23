# -*- coding: utf-8 -*-
"""
author: zengbin93
email: zeng_bin8888@163.com
create_dt: 2023/4/3 14:45
describe: 信号观察页面
"""
import os
os.environ['czsc_research_cache'] = r'./CZSC投研共享数据'
import czsc
import numpy as np
import pandas as pd
import streamlit as st
from datetime import timedelta
from czsc.utils import sorted_freqs
from czsc.connectors.research import get_raw_bars, get_symbols

st.set_page_config(layout="wide")


params = st.experimental_get_query_params()
default_signals_module = params['signals_module'][0] if 'signals_module' in params else "czsc.signals"
default_conf = params['conf'][0] if 'conf' in params else "日线_RBreaker_BS辅助V230326_做多_反转_任意_0"
default_freqs = params['freqs'] if 'freqs' in params else ['30分钟', '日线', '周线']

signals_module = st.sidebar.text_input("信号模块名称：", value=default_signals_module)
parser = czsc.SignalsParser(signals_module=signals_module)

plotly_config = {
    "scrollZoom": True,
    "displayModeBar": True,
    "displaylogo": False,
    'modeBarButtonsToRemove': [
        'toggleSpikelines',
        'select2d',
        'zoomIn2d',
        'zoomOut2d',
        'lasso2d',
        'autoScale2d',
        'hoverClosestCartesian',
        'hoverCompareCartesian']}

st.title("信号识别结果观察")

with st.sidebar:
    st.header("信号配置")
    with st.form("my_form"):
        conf = st.text_input("请输入信号：", value=default_conf)
        symbol = st.selectbox("请选择股票：", get_symbols('ALL'), index=0)
        freqs = st.multiselect("请选择周期：", sorted_freqs, default=default_freqs)
        freqs = czsc.freqs_sorted(freqs)
        sdt = st.date_input("开始日期：", value=pd.to_datetime('2022-01-01'))
        edt = st.date_input("结束日期：", value=pd.to_datetime('2023-01-01'))
        submit_button = st.form_submit_button(label='提交')


@st.cache_data(ttl=60*60*24)
def create_kline_chart(symbol, conf, freqs, sdt, edt):
    # 获取K线，计算信号
    _edt = pd.to_datetime(edt)
    _sdt = pd.to_datetime(sdt)
    bars = get_raw_bars(symbol, freqs[0], _sdt - timedelta(days=365*3), _edt)
    signals_config = czsc.get_signals_config([conf], signals_module=signals_module)
    sigs = czsc.generate_czsc_signals(bars, signals_config, df=True, sdt=_sdt)
    sigs.drop(columns=['freq', 'cache'], inplace=True)
    cols = [x for x in sigs.columns if len(x.split('_')) == 3]
    assert len(cols) == 1
    sigs['match'] = sigs.apply(czsc.Signal(conf).is_match, axis=1)
    sigs['text'] = np.where(sigs['match'], sigs[cols[0]], "")

    # 在图中绘制指定需要观察的信号
    chart = czsc.KlineChart(n_rows=3, height=800)
    chart.add_kline(sigs, freqs[0])
    chart.add_sma(sigs, row=1, ma_seq=(5, 10, 20), visible=True)
    chart.add_vol(sigs, row=2)
    chart.add_macd(sigs, row=3)
    df1 = sigs[sigs['text'] != ""][['dt', 'text', 'close', 'low']].copy()
    chart.add_scatter_indicator(x=df1['dt'], y=df1['low'], row=1, name='信号', mode='markers',
                                marker_size=20, marker_color='red', marker_symbol='triangle-up')

    # 绘制笔 + 分型
    c = czsc.CZSC([x for x in bars if _edt > x.dt > _sdt], max_bi_num=300)
    bi_list = c.bi_list
    bi1 = [{'dt': x.fx_a.dt, "bi": x.fx_a.fx, "text": x.fx_a.mark.value} for x in bi_list]
    bi2 = [{'dt': bi_list[-1].fx_b.dt, "bi": bi_list[-1].fx_b.fx, "text": bi_list[-1].fx_b.mark.value[0]}]
    bi = pd.DataFrame(bi1 + bi2)
    fx = pd.DataFrame([{'dt': x.dt, "fx": x.fx} for x in c.fx_list])
    chart.add_scatter_indicator(fx['dt'], fx['fx'], name="分型", row=1, line_width=2)
    chart.add_scatter_indicator(bi['dt'], bi['bi'], name="笔", row=1, line_width=2)
    return chart


_chart = create_kline_chart(symbol, conf, freqs, sdt, edt)
st.plotly_chart(_chart.fig, use_container_width=True, config=plotly_config)
st.experimental_set_query_params(signals_module=signals_module, conf=conf, freqs=freqs)
