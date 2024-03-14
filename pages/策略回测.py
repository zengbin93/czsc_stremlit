import czsc
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(layout="wide", page_icon="🐣", page_title="持仓权重测试")


@st.cache_data(ttl=3600 * 3)
def show_backtest(df, delay, only_direction, fee, digits):
    df['dt'] = pd.to_datetime(df['dt'])
    df = df.sort_values(["symbol", "dt"]).reset_index(drop=True)

    if delay > 0:
        for _, dfg in df.groupby("symbol"):
            df.loc[dfg.index, 'weight'] = dfg['weight'].shift(delay).fillna(0)

    if only_direction:
        df['weight'] = np.sign(df['weight'])

    czsc.show_weight_backtest(df, fee=fee, digits=digits,
                              show_yearly_stats=True, show_backtest_detail=True,
                              show_monthly_return=True, show_daily_stats=True)


def main():
    st.title('持仓权重测试')
    st.divider()

    with st.expander("上传持仓权重进行测试", expanded=True):
        with st.form(key='my_form'):
            c1, c2, c3 = st.columns([2, 1, 1])
            file = c1.file_uploader('上传文件', type=['csv', 'feather'], accept_multiple_files=False)
            fee = c2.number_input('单边费率（BP）', value=2.0, step=0.1, min_value=-100.0, max_value=100.0)
            digits = c2.number_input('小数位数', value=2, step=1, min_value=0, max_value=10)
            delay = c3.number_input("延迟执行", value=0, step=1, min_value=0, max_value=100, help="测试策略对执行是否敏感")
            only_direction = c3.selectbox("按方向测试", [False, True], index=0)
            submit_button = st.form_submit_button(label='开始测试')

    if not submit_button:
        st.stop()

    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.name.endswith('.feather'):
        df = pd.read_feather(file)
    else:
        raise ValueError(f"不支持的文件类型: {file.name}")

    show_backtest(df, delay, only_direction, fee, digits)


if __name__ == '__main__':
    main()
