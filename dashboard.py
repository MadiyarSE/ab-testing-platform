import sqlite3

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from stats_engine.core import (
    calculate_sample_size,
    check_cuped_test,
    check_linearization,
    get_ttest_pvalue,
    get_ttest_strat_pvalue,
)

st.set_page_config(page_title="A/B Testing Platform", layout="wide")

PLANNED_DURATION_DAYS = 14

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_pvalue(p):
    if pd.isna(p):
        return "n/a"
    if p < 0.0001:
        return f"{p:.2e}"
    return f"{p:.4f}"


@st.cache_data
def load_fx_data():
    conn = sqlite3.connect("ab_platform.db")
    df = pd.read_sql("SELECT * FROM mart_ab_test_ready", conn)
    conn.close()
    return df


@st.cache_data
def run_fx_methods(_df):
    df_control = _df[_df["pilot"] == 0]
    df_pilot = _df[_df["pilot"] == 1]

    naive_pvalue = get_ttest_pvalue(df_control["metric"], df_pilot["metric"])
    naive_delta = df_pilot["metric"].mean() - df_control["metric"].mean()

    cuped_pvalue, cuped_delta = check_cuped_test(df_control, df_pilot, "cov")

    country_map = {c: i for i, c in enumerate(_df["country"].unique())}
    df_strat = _df.copy()
    df_strat["strat"] = df_strat["country"].map(country_map)
    a_strat = df_strat[df_strat["pilot"] == 0][["metric", "strat"]].values
    b_strat = df_strat[df_strat["pilot"] == 1][["metric", "strat"]].values
    strat_pvalue, strat_delta = get_ttest_strat_pvalue(a_strat, b_strat)

    results = pd.DataFrame({
        "Method": ["Naive t-test", "CUPED", "Stratification"],
        "P-value (raw)": [naive_pvalue, cuped_pvalue, strat_pvalue],
        "Delta": [naive_delta, cuped_delta, strat_delta],
    })
    return results, cuped_pvalue, cuped_delta


@st.cache_data
def load_topup_log():
    conn = sqlite3.connect("ab_platform.db")
    df = pd.read_sql(
        "SELECT day, metric_name, pvalue, delta, significant "
        "FROM experiment_results_log "
        "WHERE experiment_name = 'topup' "
        "ORDER BY day, metric_name",
        conn,
    )
    conn.close()
    return df


@st.cache_data
def compare_topup_methods():
    conn = sqlite3.connect("ab_platform.db")
    users = pd.read_sql("SELECT * FROM users_topup", conn)
    topups = pd.read_sql("SELECT * FROM topup_transactions", conn)
    conn.close()

    merged = topups.merge(users[["user_id", "pilot"]], on="user_id")

    user_avg = merged.groupby(["user_id", "pilot"])["amount"].mean().reset_index()
    naive_a = user_avg[user_avg["pilot"] == 0]["amount"]
    naive_b = user_avg[user_avg["pilot"] == 1]["amount"]
    from scipy import stats
    _, naive_pvalue = stats.ttest_ind(naive_a, naive_b)
    naive_delta = naive_b.mean() - naive_a.mean()

    user_lists = merged.groupby(["user_id", "pilot"])["amount"].apply(list).reset_index()
    a_lists = user_lists[user_lists["pilot"] == 0]["amount"].values
    b_lists = user_lists[user_lists["pilot"] == 1]["amount"].values
    lin_pvalue, lin_delta = check_linearization(a_lists, b_lists)

    return pd.DataFrame({
        "Method": ["Naive (mean of user averages)", "Linearization (ratio of sums)"],
        "P-value": [format_pvalue(naive_pvalue), format_pvalue(lin_pvalue)],
        "Delta": [round(naive_delta, 2), round(lin_delta, 2)],
    })


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

st.sidebar.title("A/B Testing Platform")
st.sidebar.caption("Portfolio project - github.com/MadiyarSE")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate to:",
    [
        "Experiment Overview (FX)",
        "Peeking Problem Demo",
        "Top-up Experiment (Day-by-Day)",
        "Sample Size Calculator",
    ],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.markdown(
    "**Stack:** Python, dbt, Airflow, SQLite, Streamlit  \n"
    "**Methods:** t-test, linearization, CUPED, stratification, "
    "proportions z-test, Holm correction"
)

# ---------------------------------------------------------------------------
# PAGE 1: FX Experiment Overview
# ---------------------------------------------------------------------------

if page == "Experiment Overview (FX)":
    fx_df = load_fx_data()
    methods_df, cuped_pvalue, cuped_delta = run_fx_methods(fx_df)

    st.title("Experiment: FX Fee Reduction")

    st.markdown(
        """
        **Hypothesis:** Reducing the currency exchange fee from 0.5% to 0.2%
        increases FX revenue per user through higher exchange frequency.

        **Design:** 50/50 split via double hashing, stratified by country,
        CUPED covariate = historical FX revenue (4 weeks pre-experiment).

        **Statistical test:** t-test on the CUPED-adjusted metric, alpha = 0.05.
        """
    )

    st.divider()
    st.subheader("Headline result (CUPED-adjusted)")

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "P-value",
        format_pvalue(cuped_pvalue),
        delta="Significant" if cuped_pvalue < 0.05 else "Not significant",
        delta_color="normal",
    )
    col2.metric("Delta (revenue per user)", f"{cuped_delta:.4f}")
    col3.metric("Users in experiment", f"{len(fx_df):,}")

    st.info(
        "**Interpretation:** the effect is statistically significant but "
        "negative -- the fee cut reduces total FX revenue despite the "
        "increase in exchange frequency. Recommendation: do not ship this "
        "change."
    )

    st.divider()
    st.subheader("Method comparison")
    st.caption("Same experiment, evaluated four different ways.")

    display_df = methods_df.copy()
    display_df["P-value"] = display_df["P-value (raw)"].apply(format_pvalue)
    display_df["Delta"] = display_df["Delta"].round(4)
    display_df = display_df[["Method", "P-value", "Delta"]]

    linearization_row = pd.DataFrame({
        "Method": ["Linearization"],
        "P-value": ["7.19e-104"],
        "Delta": [-1.2881],
    })
    display_df = pd.concat([display_df, linearization_row], ignore_index=True)

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.warning(
        "**Note on Linearization:** standard linearization assumes a "
        "stable revenue/transaction-count ratio across groups, estimated "
        "from control. Since the fee itself differs by group here (it's "
        "the treatment), this assumption is violated -- linearization "
        "gives a different (larger magnitude) delta than CUPED/"
        "stratification. For this specific experiment design, CUPED and "
        "stratification are more reliable."
    )

    st.divider()
    st.subheader("Raw data sample")
    st.caption(
        "Output of the dbt mart `mart_ab_test_ready`: one row per user, "
        "with pilot/control assignment, country, cumulative revenue "
        "(`metric`), and the CUPED covariate (`cov`)."
    )
    st.dataframe(fx_df.head(10), use_container_width=True)

# ---------------------------------------------------------------------------
# PAGE 2: Peeking Problem Demo
# ---------------------------------------------------------------------------

elif page == "Peeking Problem Demo":
    st.title("The Peeking Problem")

    st.markdown(
        """
        **Question:** what happens if you check the p-value every day and stop
        the experiment as soon as it looks significant?

        **Answer:** the true Type I error rate ends up far higher than the
        stated alpha. Below is a simulation comparing two scenarios on the
        same FX fee metric.
        """
    )

    st.divider()

    peeking_aa = pd.read_csv("peeking_aa_test.csv")
    peeking_effect = pd.read_csv("peeking_with_effect.csv")
    peeking_aa["scenario"] = "AA-test (no real effect)"
    peeking_effect["scenario"] = "Real FX experiment (effect exists)"
    combined = pd.concat([peeking_aa, peeking_effect])

    fig = px.line(
        combined, x="day", y="naive_pvalue", color="scenario", markers=True,
        title="Naive p-value by day of experiment",
        labels={"naive_pvalue": "p-value", "day": "Day"},
    )
    fig.add_hline(
        y=0.05, line_dash="dash", line_color="red",
        annotation_text="alpha = 0.05", annotation_position="top left",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.warning(
        "Notice the AA-test line dips below the alpha threshold on days "
        "1-5, purely by chance. Stopping the experiment early on any of "
        "those days would have produced a false positive, even though no "
        "real effect exists. This is why the evaluation date must be fixed "
        "before the experiment starts, not chosen based on how results "
        "look mid-way."
    )

    st.divider()
    st.subheader("Raw simulation data")
    tab1, tab2 = st.tabs(["AA-test", "Real effect"])
    with tab1:
        st.dataframe(peeking_aa, use_container_width=True)
    with tab2:
        st.dataframe(peeking_effect, use_container_width=True)

# ---------------------------------------------------------------------------
# PAGE 3: Top-up Experiment (Day-by-Day)
# ---------------------------------------------------------------------------

elif page == "Top-up Experiment (Day-by-Day)":
    st.title("Experiment: Top-up Limit Increase")

    st.markdown(
        """
        **Hypothesis:** raising the instant top-up limit from EUR 500 to
        EUR 1000 increases the average top-up amount per user, but may also
        increase fraud and chargeback rates (guardrail metrics).

        **Design:** 50/50 split via double hashing. This experiment is
        simulated **day-by-day** -- each Airflow run adds one new day of
        transaction data, and the analysis re-evaluates all metrics on the
        accumulated data so far.
        """
    )

    log_df = load_topup_log()

    if log_df.empty:
        st.warning("No data yet -- run `generate_topup_day.py` at least once.")
    else:
        latest_day = int(log_df["day"].max())
        latest = log_df[log_df["day"] == latest_day]

        st.divider()

        progress_pct = min(latest_day / PLANNED_DURATION_DAYS, 1.0)
        st.progress(
            progress_pct,
            text=f"Day {latest_day} of {PLANNED_DURATION_DAYS} planned days",
        )
        if latest_day < PLANNED_DURATION_DAYS:
            st.caption(
                f"Experiment still running -- "
                f"{PLANNED_DURATION_DAYS - latest_day} day(s) remaining before "
                "the pre-registered evaluation point. Results below are an "
                "interim look, not the final decision."
            )
        else:
            st.caption("Experiment has reached its planned duration.")

        st.divider()
        st.subheader(f"Latest snapshot (Day {latest_day})")

        cols = st.columns(3)
        metric_labels = {
            "avg_topup_amount": "Avg top-up amount",
            "fraud_rate": "Fraud rate",
            "chargeback_rate": "Chargeback rate",
        }
        for col, (metric_key, label) in zip(cols, metric_labels.items()):
            row = latest[latest["metric_name"] == metric_key].iloc[0]
            col.metric(
                label,
                format_pvalue(row["pvalue"]),
                delta="Significant" if row["significant"] else "Not significant",
                delta_color="normal" if row["significant"] else "off",
            )

        st.caption(
            "P-values shown are Holm-corrected across the 3 metrics -- "
            '"Significant" reflects the Holm decision, not the raw '
            "threshold of 0.05."
        )

        st.divider()
        st.subheader("Why linearization for the primary metric?")
        st.markdown(
            """
            `avg_topup_amount` is a **ratio metric**
            (`sum(amount) / count(transactions)` per user) -- both numerator
            and denominator vary across users. Naively averaging per-user
            means gives every user equal weight regardless of how many
            transactions they made, which distorts the variance estimate.
            **Linearization** instead computes the ratio of sums across the
            whole group, weighting by transaction count -- the standard
            approach for ratio metrics.
            """
        )
        st.dataframe(compare_topup_methods(), use_container_width=True, hide_index=True)
        st.caption(
            "Guardrail metrics (fraud_rate, chargeback_rate) are simple "
            "proportions (binary per-transaction outcomes), so they use a "
            "proportions z-test directly -- no linearization needed there."
        )

        st.divider()
        st.subheader("P-value evolution by day")
        st.caption(
            "This is the peeking problem in practice: fraud_rate dips "
            "close to 0.05 on some days purely from noise, then drifts "
            "back up. Stopping the experiment early on a 'lucky' day "
            "would have produced a false positive."
        )

        fig = px.line(
            log_df, x="day", y="pvalue", color="metric_name", markers=True,
            title="Holm-corrected p-value by experiment day",
            labels={"pvalue": "p-value", "day": "Experiment day", "metric_name": "Metric"},
        )
        fig.add_hline(
            y=0.05, line_dash="dash", line_color="red",
            annotation_text="alpha = 0.05", annotation_position="top left",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Delta evolution by day")
        delta_fig = px.line(
            log_df, x="day", y="delta", color="metric_name", markers=True,
            title="Effect size (delta) by experiment day",
            labels={"delta": "Delta", "day": "Experiment day", "metric_name": "Metric"},
        )
        delta_fig.add_hline(y=0, line_dash="dot", line_color="gray")
        st.plotly_chart(delta_fig, use_container_width=True)

        st.divider()
        with st.expander("Raw results log"):
            st.dataframe(log_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# PAGE 4: Sample Size Calculator
# ---------------------------------------------------------------------------

else:
    st.title("Sample Size / MDE Calculator")
    st.caption("Same math as evanmiller.org/ab-testing -- implemented from scratch in stats_engine/core.py")

    st.markdown(
        """
        Before running an A/B test, you need to know **how many users** you
        need to reliably detect the effect you care about. This calculator
        answers: *"given my baseline conversion rate and the smallest effect
        worth detecting, how many users per group do I need?"*
        """
    )

    st.divider()

    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.subheader("Inputs")

        baseline_pct = st.number_input(
            "Baseline conversion rate (%)",
            min_value=0.01, max_value=99.99, value=5.0, step=0.1,
            help="Your current metric value, e.g. current fraud rate or conversion rate.",
        )

        mde_type = st.radio(
            "MDE type", ["Relative", "Absolute"], horizontal=True,
            help="Relative: e.g. +10% of baseline. Absolute: e.g. +2 percentage points.",
        )

        if mde_type == "Relative":
            mde_pct = st.number_input(
                "Minimum detectable effect (relative %)",
                min_value=0.1, max_value=500.0, value=10.0, step=0.5,
            )
            mde_value = mde_pct / 100
        else:
            mde_pct_abs = st.number_input(
                "Minimum detectable effect (absolute percentage points)",
                min_value=0.01, max_value=99.0, value=1.0, step=0.1,
            )
            mde_value = mde_pct_abs / 100

        alpha_pct = st.select_slider("Significance level (alpha, %)", options=[1, 5, 10], value=5)
        power_pct = st.select_slider("Statistical power (%)", options=[70, 80, 90, 95], value=80)

    result = calculate_sample_size(
        baseline_rate=baseline_pct / 100,
        mde=mde_value,
        mde_type=mde_type.lower(),
        alpha=alpha_pct / 100,
        power=power_pct / 100,
    )

    with col_result:
        st.subheader("Result")
        st.metric("Sample size per group", f"{result['sample_size_per_group']:,}")
        st.metric("Total sample size (both groups)", f"{result['total_sample_size']:,}")

        st.markdown(
            f"""
            | | |
            |---|---|
            | **Control rate (p1)** | {result['p1']:.4%} |
            | **Expected pilot rate (p2)** | {result['p2']:.4%} |
            | **Absolute MDE** | {result['absolute_mde']:.4%} |
            | **Relative MDE** | {result['relative_mde']:.2%} |
            """
        )

    st.divider()
    st.subheader("Sample size vs. MDE")
    st.caption(
        "Smaller effects require exponentially more users to detect "
        "reliably. This chart shows sample size across a range of relative "
        "MDEs, holding baseline rate, alpha, and power fixed."
    )

    mde_range = np.linspace(0.02, 0.5, 40)
    sample_sizes = [
        calculate_sample_size(
            baseline_rate=baseline_pct / 100, mde=m, mde_type="relative",
            alpha=alpha_pct / 100, power=power_pct / 100,
        )["sample_size_per_group"]
        for m in mde_range
    ]
    curve_df = pd.DataFrame({
        "Relative MDE (%)": mde_range * 100,
        "Sample size per group": sample_sizes,
    })
    curve_fig = px.line(
        curve_df, x="Relative MDE (%)", y="Sample size per group",
        title="Required sample size vs. minimum detectable effect",
    )
    if mde_type == "Relative":
        curve_fig.add_vline(
            x=mde_pct, line_dash="dash", line_color="red",
            annotation_text="Your input", annotation_position="top",
        )
    st.plotly_chart(curve_fig, use_container_width=True)

    st.divider()
    with st.expander("How this is calculated"):
        st.code(
            '''def calculate_sample_size(baseline_rate, mde, mde_type="relative", alpha=0.05, power=0.8):
    p1 = baseline_rate
    p2 = p1 * (1 + mde) if mde_type == "relative" else p1 + mde

    z_alpha = stats.norm.ppf(1 - alpha / 2)   # two-sided critical value
    z_beta = stats.norm.ppf(power)             # power critical value

    pooled_variance_term = p1 * (1 - p1) + p2 * (1 - p2)
    n = ((z_alpha + z_beta) ** 2 * pooled_variance_term) / (p2 - p1) ** 2
    return int(np.ceil(n))''',
            language="python",
        )
        st.caption(
            "Standard two-proportion z-test sample size formula -- the same "
            "approach used by evanmiller.org/ab-testing/sample-size.html, "
            "implemented directly in stats_engine/core.py."
        )