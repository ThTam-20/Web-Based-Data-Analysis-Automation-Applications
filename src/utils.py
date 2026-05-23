import matplotlib
matplotlib.use("Agg")  # Non-interactive backend cho Streamlit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st


def execute_plt_code(code: str, df: pd.DataFrame):
    """Execute the passing code to plot figure

    Args:
        code (str): action string (containing plt code)
        df (pd.DataFrame): our dataframe

    Returns:
        _type_: plt figure
    """

    try:
        # Đóng tất cả figure cũ trước khi vẽ mới
        plt.close("all")

        local_vars = {
            "plt": plt,
            "df": df,
            "pd": pd,
            "np": np,
            "sns": sns,
        }
        compiled_code = compile(code, "<string>", "exec")
        exec(compiled_code, {}, local_vars)

        return plt.gcf()

    except Exception as e:
        st.error(f"Error executing plt code: {e}")
        return None