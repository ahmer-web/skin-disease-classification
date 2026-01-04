import streamlit as st
from scripts.predict import predict_image, predict_image_bytes, model, target_layer
from scripts.gradcam import generate_gradcam
from scripts.auth import (
    init_db as init_user_db,
    register_user,
    login_user,
    login_session,
    logout_session,
)
from scripts.history_db import (
    init_db as init_hist_db,
    log_prediction,
    get_predictions_for_user,
    get_all_predictions,
    delete_prediction,
    clear_history_for_user,
    clear_all_history,
)
import matplotlib.pyplot as plt
import pandas as pd

# -------------------------------------------------
# ✅ Init DBs & Session State
# -------------------------------------------------
init_user_db()
init_hist_db()

st.set_page_config(
    page_title="Skin Disease Classifier",
    layout="wide"
)

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
    st.session_state["username"] = None
    st.session_state["is_admin"] = False


def risk_label(conf_pct: float) -> str:
    if conf_pct >= 80:
        return "High"
    elif conf_pct >= 50:
        return "Moderate"
    else:
        return "Low"


# -------------------------------------------------
# ✅ Login / Signup UI (shown if NOT logged in)
# -------------------------------------------------
def show_auth_page():
    st.markdown("<h1 style='text-align: center;'>🩺 Skin Disease Classification System</h1>", unsafe_allow_html=True)
    st.write("<p style='text-align: center;'>Please log in or create an account to use the system.</p>", unsafe_allow_html=True)

    tab_login, tab_signup = st.tabs(["🔐 Login", "🧾 Sign Up"])

    with tab_login:
        st.subheader("Login")
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            ok, is_admin = login_user(username, password)
            if ok:
                login_session(username, is_admin)
                st.success(f"Logged in as {username} ({'Admin' if is_admin else 'User'})")
                st.rerun()
            else:
                st.error("Invalid username or password.")

    with tab_signup:
        st.subheader("Create New Account")
        new_user = st.text_input("Choose a username", key="signup_user")
        new_pass = st.text_input("Choose a password", type="password", key="signup_pass")
        if st.button("Sign Up"):
            success, is_admin, err = register_user(new_user, new_pass)
            if success:
                st.success("Account created successfully! You can now log in.")
                if is_admin:
                    st.info("You are the first user, so you have admin rights.")
            else:
                st.error(err or "Registration failed.")


# -------------------------------------------------
# ✅ Main App (shown when logged in)
# -------------------------------------------------
def show_main_app():
    st.markdown("<h1 style='text-align: center;'>🩺 Skin Disease Classification System</h1>", unsafe_allow_html=True)
    st.write("<p style='text-align: center;'>Upload dermoscopic skin lesion images to get predictions and visual explanations.</p>", unsafe_allow_html=True)

    # Top bar with user info + logout
    col_user, col_logout = st.columns([4, 1])
    with col_user:
        role = "Admin" if st.session_state["is_admin"] else "User"
        st.markdown(f"**Logged in as:** `{st.session_state['username']}`  &nbsp;&nbsp; _(Role: {role})_")
    with col_logout:
        if st.button("Logout"):
            logout_session()
            st.rerun()

    # -------------------------------------------------
    # ✅ Tabs – admin gets an extra Analytics tab
    # -------------------------------------------------
    if st.session_state["is_admin"]:
        tab_labels = [
            "🔍 Single Image (with Grad-CAM)",
            "📂 Batch Prediction (Multiple Images)",
            "📜 Prediction History",
            "📈 Admin Analytics",
        ]
    else:
        tab_labels = [
            "🔍 Single Image (with Grad-CAM)",
            "📂 Batch Prediction (Multiple Images)",
            "📜 Prediction History",
        ]

    tabs = st.tabs(tab_labels)
    tab_single = tabs[0]
    tab_batch = tabs[1]
    tab_history = tabs[2]
    tab_analytics = tabs[3] if st.session_state["is_admin"] else None

    # ============================================================
    # ✅ TAB 1: SINGLE IMAGE + GRAD-CAM
    # ============================================================
    with tab_single:
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"],
            key="single_uploader"
        )

        if uploaded_file is not None:
            image, results, img_tensor, tgt_layer = predict_image(uploaded_file)
            top_class, top_conf = results[0]
            conf_pct = round(top_conf * 100, 2)
            r_label = risk_label(conf_pct)

            col1, col2 = st.columns([1, 1])

            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)

            with col2:
                st.markdown("### 🔬 Model Prediction:")
                st.success(f"✅ **Predicted Class: {top_class}**")
                st.markdown(f"### Confidence: **{conf_pct:.2f}%**")
                st.markdown(f"**Risk Level:** {r_label}")

            # Log prediction
            log_prediction(
                username=st.session_state["username"],
                filename=uploaded_file.name,
                mode="single",
                predicted_class=top_class,
                confidence=conf_pct,
                risk_level=r_label,
            )

            # Grad-CAM
            heatmap_img, overlay_img = generate_gradcam(img_tensor, model, tgt_layer)

            st.markdown("---")
            st.markdown("### 🔥 Grad-CAM Visual Explanation")

            cam1, cam2 = st.columns([1, 1])
            with cam1:
                st.image(heatmap_img, caption="Grad-CAM Heatmap", use_container_width=True)
            with cam2:
                st.image(overlay_img, caption="Heatmap Overlay", use_container_width=True)

            # Confidence chart
            st.markdown("---")
            st.markdown("### 📊 Confidence Breakdown:")

            labels = [r[0] for r in results]
            scores = [r[1] for r in results]

            fig, ax = plt.subplots(figsize=(8, 3))
            ax.bar(labels, scores)
            ax.set_ylim([0, 1])
            plt.xticks(rotation=45)
            st.pyplot(fig)

            st.markdown(
                "<div style='margin-top: 20px; padding: 10px; background-color: #1E1E1E; border-radius: 8px; text-align: center;'>"
                "<span style='color:#F5C542;'>⚠ This tool is for educational and research purposes only and is not a substitute for professional medical diagnosis.</span>"
                "</div>",
                unsafe_allow_html=True
            )
        else:
            st.info("⬆ Upload an image to begin classification.")

    # ============================================================
    # ✅ TAB 2: BATCH PREDICTION (Enhanced)
    # ============================================================
    with tab_batch:
        st.write("Upload multiple images to get predictions in an enhanced results layout.")

        batch_files = st.file_uploader(
            "Choose one or more images...",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="batch_uploader"
        )

        if batch_files:
            batch_results = []

            for file in batch_files:
                image, top_class, top_conf, _ = predict_image_bytes(file)
                conf_pct = round(top_conf * 100, 2)
                r_label = risk_label(conf_pct)

                batch_results.append({
                    "file": file.name,
                    "image": image,
                    "pred": top_class,
                    "conf": conf_pct,
                    "risk": r_label,
                })

                log_prediction(
                    username=st.session_state["username"],
                    filename=file.name,
                    mode="batch",
                    predicted_class=top_class,
                    confidence=conf_pct,
                    risk_level=r_label,
                )

            batch_results = sorted(batch_results, key=lambda x: x["conf"], reverse=True)

            st.markdown("### 🖼 Batch Classification Results")

            for item in batch_results:
                col1, col2, col3 = st.columns([1, 2, 2])

                with col1:
                    st.image(item["image"], width=120)

                with col2:
                    st.markdown(f"**File:** {item['file']}")
                    st.markdown(f"**Prediction:** `{item['pred']}`")

                with col3:
                    conf = item["conf"]
                    label = item["risk"]
                    if label == "High":
                        st.success(f"✅ {conf}% (High Confidence)")
                    elif label == "Moderate":
                        st.warning(f"🟡 {conf}% (Moderate Confidence)")
                    else:
                        st.error(f"🔴 {conf}% (Low — Needs Dermatologist Review)")

                st.markdown("---")

            df = pd.DataFrame([{
                "File Name": i["file"],
                "Predicted Class": i["pred"],
                "Confidence (%)": i["conf"],
                "Risk Level": i["risk"],
            } for i in batch_results])

            st.download_button(
                label="⬇ Download results as CSV",
                data=df.to_csv(index=False),
                file_name="batch_predictions.csv",
                mime="text/csv"
            )

        else:
            st.info("⬆ Upload multiple images to run batch prediction.")

    # ============================================================
    # ✅ TAB 3: HISTORY (Per-User / Admin with controls)
    # ============================================================
    with tab_history:
        st.write("View recent predictions made by the system.")

        if st.session_state["is_admin"]:
            st.info("You are an admin. Showing history for ALL users.")
            rows = get_all_predictions(limit=500)
        else:
            st.info("Showing your own prediction history.")
            rows = get_predictions_for_user(st.session_state["username"], limit=200)

        if rows:
            df_hist = pd.DataFrame(
                rows,
                columns=[
                    "ID",
                    "Timestamp",
                    "Username",
                    "File Name",
                    "Mode",
                    "Predicted Class",
                    "Confidence (%)",
                    "Risk Level",
                ],
            )

            st.dataframe(df_hist, use_container_width=True)

            st.download_button(
                label="⬇ Download history as CSV",
                data=df_hist.to_csv(index=False),
                file_name="prediction_history.csv",
                mime="text/csv"
            )

            if st.session_state["is_admin"]:
                st.markdown("### 🛠 Admin Controls")

                col1, col2, col3 = st.columns(3)

                with col1:
                    rec_id = st.number_input(
                        "Record ID to delete",
                        min_value=1,
                        step=1,
                        format="%d",
                    )
                    if st.button("Delete record"):
                        delete_prediction(int(rec_id))
                        st.success(f"Deleted record ID {int(rec_id)}")
                        st.rerun()

                with col2:
                    user_to_clear = st.text_input("Username to clear history for")
                    if st.button("Clear this user's history"):
                        if user_to_clear.strip():
                            clear_history_for_user(user_to_clear.strip())
                            st.success(f"Cleared history for '{user_to_clear.strip()}'")
                            st.rerun()
                        else:
                            st.warning("Please enter a username.")

                with col3:
                    if st.button("⚠ Clear ALL history"):
                        clear_all_history()
                        st.warning("All prediction history has been cleared.")
                        st.rerun()

        else:
            st.info("No predictions logged yet. Make some predictions in the other tabs first.")

# ============================================================
# ✅ TAB 4: ADMIN ANALYTICS (Admin only)
# ============================================================
    if st.session_state["is_admin"]:
     with tab_analytics:

        st.markdown("## 📊 System Analytics Dashboard")
        st.write("Analytics based on all stored predictions across all users.")

        rows = get_all_predictions(limit=2000)

        if not rows:
            st.info("No prediction data available yet.")
        else:
            df = pd.DataFrame(
                rows,
                columns=[
                    "ID",
                    "Timestamp",
                    "Username",
                    "File Name",
                    "Mode",
                    "Predicted Class",
                    "Confidence (%)",
                    "Risk Level",
                ],
            )

            # Parse timestamps
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

            # ==========================
            # ✅ KPIs / metrics
            # ==========================
            total_preds = len(df)
            high_count = (df["Risk Level"] == "High").sum()
            mod_count = (df["Risk Level"] == "Moderate").sum()
            low_count = (df["Risk Level"] == "Low").sum()

            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Total Predictions", total_preds)
            kpi2.metric("High Risk", high_count)
            kpi3.metric("Moderate Risk", mod_count)
            kpi4.metric("Low Risk", low_count)

            st.markdown("---")

            # ==========================
            # ✅ Two Column Charts Layout
            # ==========================
            col_left, col_right = st.columns(2)

            # ---- Chart 1 ----
            with col_left:
                st.markdown("### 🦠 Predictions Per Disease Class")
                class_counts = df["Predicted Class"].value_counts()
                fig1, ax1 = plt.subplots(figsize=(3.5, 2))
                ax1.bar(class_counts.index, class_counts.values)
                ax1.tick_params(labelrotation=30)
                plt.tight_layout()
                st.pyplot(fig1, use_container_width=False)

            # ---- Chart 2 ----
            with col_right:
                st.markdown("### 🎯 Confidence Distribution")
                fig2, ax2 = plt.subplots(figsize=(3.5, 2))
                ax2.hist(df["Confidence (%)"], bins=10)
                ax2.set_xlabel("Confidence (%)")
                ax2.set_ylabel("Count")
                plt.tight_layout()
                st.pyplot(fig2, use_container_width=False)

            # ==========================
            # ✅ Second Row of Charts
            # ==========================
            col_left2, col_right2 = st.columns(2)

            # ---- Chart 3 ----
            with col_left2:
                st.markdown("### ⏱ Predictions Over Time")
                if df["Timestamp"].notna().any():
                    date_counts = df.groupby(df["Timestamp"].dt.date).size()
                    fig3, ax3 = plt.subplots(figsize=(3.5, 2))
                    ax3.plot(date_counts.index, date_counts.values)
                    ax3.tick_params(labelrotation=30)
                    plt.tight_layout()
                    st.pyplot(fig3, use_container_width=False)
                else:
                    st.info("Not enough timestamp data")

            # ---- Chart 4 ----
            with col_right2:
                st.markdown("### 👥 User Activity")
                user_counts = df["Username"].value_counts()
                fig4, ax4 = plt.subplots(figsize=(3.5, 2))
                ax4.bar(user_counts.index, user_counts.values)
                ax4.tick_params(labelrotation=30)
                plt.tight_layout()
                st.pyplot(fig4, use_container_width=False)

            st.markdown("---")
            st.success("✅ Analytics displayed in compact professional layout.")

# -------------------------------------------------
# ✅ App Entry Point
# -------------------------------------------------
if not st.session_state["logged_in"]:
    show_auth_page()
else:
    show_main_app()
