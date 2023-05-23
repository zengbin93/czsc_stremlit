# -*- coding: utf-8 -*-
"""
author: zengbin93
email: zeng_bin8888@163.com
create_dt: 2023/4/1 19:42
describe: 信号解析页面
"""
import czsc
import streamlit as st

st.set_page_config(layout="wide")

signals_module = st.sidebar.text_input("信号模块名称：", value="czsc.signals")
parser = czsc.SignalsParser(signals_module=signals_module)

tabs = st.tabs(['信号函数', '信号转配置', '配置转信号KEY'])


with tabs[0]:
    with st.expander("信号列表："):
        st.write(parser.sig_name_map)

    with st.expander("参数模板：", expanded=True):
        st.write(parser.sig_pats_map)


with tabs[1]:
    signals_seq = st.text_input("请输入信号：", value="日线_D1#SMA#5_BS3辅助V230319_三卖_均线新低_任意_0")
    res = parser.parse([signals_seq])[0]
    st.write("配置：", res)
    if res and res['name'].startswith("czsc.signals"):
        st.write(f"文档：https://czsc.readthedocs.io/en/latest/api/{res['name']}.html")


with tabs[2]:
    conf_example = {"freq": "日线", "di": 1, "ma_type": "SMA", "timeperiod": "5",
                    "name": "czsc.signals.cxt_third_bs_V230319"}
    conf = st.text_input("请输入配置：", value=f"{conf_example}")
    st.write("信号：", parser.config_to_keys([eval(conf)]))
