import streamlit as st
import sys
import os
import db_utils

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Feedback & Wishlist", page_icon="💬", layout="wide")

if st.session_state.get("authentication_status") is not True:
    st.warning("⚠️ Please return to the Home page and log in to access this feature.")
    st.stop()

username = st.session_state.get("username", "unknown")

db_utils.init_feedback_table()

st.title("💬 Feedback & Wishlist")
st.markdown("Every piece of feedback is read carefully and helps make this tool better. All submissions are displayed anonymously.")
st.divider()

# ══════════════════════════════════════════════════════
# Section 1: Improvement Suggestions
# ══════════════════════════════════════════════════════
st.subheader("🔧 Section 1: Improvement Suggestions")
st.caption("If you think a feature could be improved, let us know")

with st.form("form_improvement", clear_on_submit=True):
    improvement_text = st.text_area(
        "Enter your suggestion",
        placeholder="e.g. In the LCL filter design module, it would be great to export the calculation report as a PDF...",
        height=120,
        max_chars=500,
    )
    submitted_improvement = st.form_submit_button("📤 Submit Suggestion", use_container_width=True)

if submitted_improvement:
    if len(improvement_text.strip()) < 5:
        st.warning("Your suggestion is too short — please provide a bit more detail.")
    else:
        db_utils.submit_feedback(username, "improvement", improvement_text.strip())
        st.success("✅ Submitted! Thank you for your feedback.")

st.markdown("#### 📋 Submitted Suggestions")
improvement_list = db_utils.get_feedback("improvement")
if improvement_list:
    for item in improvement_list:
        with st.container(border=True):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"🙋 **{item['display_name']}**")
                st.write(item["content"])
            with col2:
                st.caption(f"🕐 {item['created_at']}")
else:
    st.info("No suggestions yet — be the first to leave one!")

st.divider()

# ══════════════════════════════════════════════════════
# Section 2: Feature Wishlist
# ══════════════════════════════════════════════════════
st.subheader("🌟 Section 2: Feature Wishlist")
st.caption("What would you like this tool to do in the future? Feel free to dream big.")

with st.form("form_wishlist", clear_on_submit=True):
    wishlist_text = st.text_area(
        "Enter your wish",
        placeholder="e.g. I'd love to see a dead-time calculation tool for three-phase inverters in a future update...",
        height=120,
        max_chars=500,
    )
    submitted_wishlist = st.form_submit_button("🌠 Submit Wishlist Item", use_container_width=True)

if submitted_wishlist:
    if len(wishlist_text.strip()) < 5:
        st.warning("Please provide a bit more detail.")
    else:
        db_utils.submit_feedback(username, "wishlist", wishlist_text.strip())
        st.success("✅ Submitted! Looking forward to building this with you.")

st.markdown("#### 📋 Community Wishlist")
wishlist_list = db_utils.get_feedback("wishlist")
if wishlist_list:
    for item in wishlist_list:
        with st.container(border=True):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"🙋 **{item['display_name']}**")
                st.write(item["content"])
            with col2:
                st.caption(f"🕐 {item['created_at']}")
else:
    st.info("No wishlist items yet — be the first to share your ideas!")
