import streamlit as st
import streamlit_authenticator as stauth
import db_utils
import time
import uuid

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="PowerTool — LCL Filter Design",
    page_icon="⚡",
    layout="centered"
)

# Initialize database (also inserts free trial code if not exists)
db_utils.init_db()

# Load credentials
credentials = db_utils.get_all_users_for_auth()

# Initialize authenticator
authenticator = stauth.Authenticate(
    credentials,
    'lcl_cookie_v2',
    'abcdef',
    30
)

# ── Page title ────────────────────────────────────────────
st.title("⚡ LCL Filter Automated Design Platform")
st.markdown("Built for **power electronics hardware engineers** · Full workflow from parameter design to hardware implementation")

# ── Tabs ─────────────────────────────────────────────────
tab_trial, tab_about, tab_buy, tab_activate, tab_login = st.tabs([
    "🎁 Free Trial",
    "📖 About",
    "💳 Purchase",
    "🚀 Activate Account",
    "🔐 Login",
])

# ──────────────────────────────────────────────────────────
# Tab 1: Free Trial
# ──────────────────────────────────────────────────────────
with tab_trial:
    st.markdown("### 🎁 Try Before You Register")
    st.markdown(
        "No registration needed. Try the full workflow — all inputs, formulas, and SOA feasibility maps "
        "are accessible. Detailed calculation results require a paid account."
    )
    st.markdown(
        """
        ✅ Experience the complete parameter input & interactive workflow
        ✅ View all theoretical formulas & SOA feasibility maps
        🔒 Recommended parameters, calculation results & design reports require paid access
        """
    )
    if st.button("▶ Start Free Trial", use_container_width=True, type="primary"):
        st.session_state["authentication_status"] = "trial"
        st.session_state["name"] = "Trial User"
        st.session_state["username"] = "__trial__"
        st.session_state["session_token"] = "__trial__"
        try:
            st.switch_page("pages/01_LCL_Filter_Design.py")
        except Exception:
            st.success("✅ Trial mode activated! Click 'LCL Filter Design' in the sidebar.")

# ──────────────────────────────────────────────────────────
# Tab 2: About
# ──────────────────────────────────────────────────────────
with tab_about:

    st.markdown("### 💡 Does this sound familiar?")
    st.markdown(
        """
        <div style='background-color:#fff1f0; border-left:4px solid #ff4d4f;
                    padding:16px; border-radius:8px; margin-bottom:12px;'>
            <p style='margin:6px 0;'>❌ Manual LCL parameter calculation takes hours — change one value and everything needs to be recalculated from scratch</p>
            <p style='margin:6px 0;'>❌ Five coupled constraints make it impossible to visualize the full feasible design space at once</p>
            <p style='margin:6px 0;'>❌ You added a passive filter, but capacitors still blow up unexpectedly on-site</p>
            <p style='margin:6px 0;'>❌ Online calculators ignore real-world factors like weak grids and transformer leakage inductance</p>
        </div>
        <div style='background-color:#f6ffed; border-left:4px solid #52c41a;
                    padding:16px; border-radius:8px; margin-bottom:8px;'>
            <p style='margin:4px 0;'>✅ This tool automates all of the above — enter your system parameters and get a complete, verified design instantly</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### 👋 Who built this?")
    st.markdown(
        """
        I'm a hardware engineer working on wind power converters. I built this tool to solve my own
        LCL filter design problems — and I'm sharing it with a **free trial** so you can experience
        the full workflow before deciding.

        Try it first, no registration needed. When you're ready, grab a license and unlock everything.

        I'd love your feedback on what works and what's missing.
        """
    )

    st.markdown("---")

    st.markdown("### 📦 10 Modules — Complete Design Workflow")
    st.caption("Parameters are linked across all modules — enter once, use everywhere.")

    modules = [
        {
            "icon": "📖",
            "title": "1. Theory Fundamentals",
            "desc": "Interactive Bode plots comparing L-filter vs LCL-filter attenuation. LaTeX-rendered transfer function derivations to understand exactly why LCL outperforms L — and when it doesn't.",
            "tag": "Theory",
            "hot": False
        },
        {
            "icon": "🛠️",
            "title": "2. LCL Parameter Design  ← Core Module",
            "desc": "Plots all five constraints — ripple, voltage drop, reactive power, and resonance safety island — simultaneously on a single SOA feasibility map. Safe zone in green, design point updates in real time, red alert on violation. Supports transformer leakage compensation and CL topology switching.",
            "tag": "🔥 Core",
            "hot": True
        },
        {
            "icon": "📈",
            "title": "3. Resonance Frequency Analysis",
            "desc": "Reveals the real impact of weak grids and transformer leakage on resonance frequency. Enter SCR or transformer uk% — auto-calculates leakage inductance and updates resonance. Supports CL/LCL topology comparison.",
            "tag": "Weak Grid",
            "hot": False
        },
        {
            "icon": "🧭",
            "title": "4. Four-Quadrant Operation",
            "desc": "Vector diagram of grid-connected current across all four quadrants. Understand the relationship between power factor angle, inductor voltage drop, and converter output voltage — with both IEEE 1547 and IEC 61400 conventions.",
            "tag": "Vector Analysis",
            "hot": False
        },
        {
            "icon": "📊",
            "title": "5. Modulation Index & Voltage Utilization",
            "desc": "Auto-calculates modulation index and voltage utilization for SPWM/SVPWM under 2-level/3-level topologies. Detects over-modulation and links back to parameter design.",
            "tag": "Modulation",
            "hot": False
        },
        {
            "icon": "🌊",
            "title": "6. Ripple Current Analysis",
            "desc": "Automatically selects ripple coefficient K based on topology and modulation strategy. Calculates maximum high-frequency ripple current in L1 with waveform visualization.",
            "tag": "Ripple",
            "hot": False
        },
        {
            "icon": "✅",
            "title": "7. Harmonic Standard Verification",
            "desc": "Built-in IEEE 519-2022 complete limit table. Auto-matches TDD and individual harmonic limits based on SCR. FFT spectrum simulation with per-harmonic compliance — non-compliant harmonics highlighted in red.",
            "tag": "IEEE 519",
            "hot": False
        },
        {
            "icon": "🎯",
            "title": "8. Passive Harmonic Filter Design",
            "desc": "Designs single-tuned passive filters for low-order harmonic violations. Automatically detects parallel anti-resonance peaks — the real reason capacitors fail on-site. Includes robustness sweep and IEEE 1531 compliance verification.",
            "tag": "⚠️ Anti-Resonance",
            "hot": True
        },
        {
            "icon": "⚡",
            "title": "9. Magnetic Component Design",
            "desc": "From parameters to physical hardware. Auto-calculates core selection, turns count, and air gap based on L1/L2 inductance and current — all steps needed for engineering implementation.",
            "tag": "Implementation",
            "hot": False
        },
        {
            "icon": "🔌",
            "title": "10. Filter Capacitor Design",
            "desc": "Auto-calculates capacitor selection from Cf, voltage rating, and resonance frequency. Accounts for tolerance and temperature drift. Outputs procurement-ready spec recommendations with datasheet verification.",
            "tag": "Capacitor Selection",
            "hot": False
        },
    ]

    for mod in modules:
        tag_color = "#ff4d4f" if mod["hot"] else "#0d6efd"
        st.markdown(
            f"""
            <div style='border:1px solid #e8e8e8; border-radius:10px; padding:16px;
                        margin-bottom:12px; background:#fafafa;'>
                <div style='display:flex; align-items:center; gap:10px; margin-bottom:6px;'>
                    <span style='font-size:22px;'>{mod["icon"]}</span>
                    <strong style='font-size:16px;'>{mod["title"]}</strong>
                    <span style='background:{tag_color}; color:white; font-size:11px;
                                 padding:2px 8px; border-radius:10px; margin-left:auto;
                                 white-space:nowrap;'>{mod["tag"]}</span>
                </div>
                <p style='margin:0; color:#555; font-size:14px; line-height:1.6;'>{mod["desc"]}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown(
        """
        <div style='background: linear-gradient(135deg, #fdf4ff 0%, #f3e8ff 100%);
                    border:1px solid #a855f7; border-radius:12px; padding:20px;
                    text-align:center;'>
            <h3 style='margin:0 0 8px 0; color:#7e22ce;'>💳 Ready to unlock everything?</h3>
            <p style='color:#555; font-size:15px; margin:0 0 8px 0;'>
                One-time purchase · Lifetime access · All future updates included
            </p>
            <p style='color:#555; font-size:14px; margin:0;'>
                👆 Click <strong>"Purchase"</strong> tab above to get started.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ──────────────────────────────────────────────────────────
# Tab 3: Purchase
# ──────────────────────────────────────────────────────────
with tab_buy:
    st.markdown("### 💳 Purchase License")
    st.markdown("""
    <div style='background: linear-gradient(135deg, #f0f4ff 0%, #fafaff 100%);
                border:1px solid #c7d2fe; border-radius:12px; padding:20px;
                text-align:center; margin-bottom:20px;'>
        <p style='font-size:36px; font-weight:bold; color:#4f46e5; margin:0;'>$29</p>
        <p style='color:#888; font-size:13px; margin:8px 0 0 0;'>
            One-time purchase · Lifetime access · All future updates included
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("#### How to get full access")
    st.markdown("""
    1. Click the button below to go to our secure checkout page
    2. Complete payment (credit card or PayPal accepted)
    3. You will receive a **license key** via email automatically
    4. Come back here, click the **Activate Account** tab, and enter your license key to register
    """)
    st.markdown("""
    <div style='text-align:center; margin: 28px 0 16px 0;'>
        <a href='https://www.creem.io/test/payment/prod_JKYmRRxAf2SFRP29lgu4g'
           target='_blank'
           style='display:inline-block; background:#a855f7; color:white;
                  padding:14px 40px; border-radius:10px; text-decoration:none;
                  font-size:17px; font-weight:bold; box-shadow: 0 4px 12px rgba(168,85,247,0.3);'>
            🛒 Purchase Now — $29
        </a>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#888; font-size:13px;'>Questions? Contact gpy1090@gmail.com</p>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
# Tab 4: Activate Account
# ──────────────────────────────────────────────────────────
with tab_activate:
    st.markdown("### 🚀 Activate Your Account")
    st.markdown(
        "Already purchased? Enter your license key below to create your account and unlock all modules."
    )

    with st.form("register_form"):
        new_username        = st.text_input("Username", placeholder="e.g. john_doe")
        new_name            = st.text_input("Your name", placeholder="e.g. John")
        new_email           = st.text_input("Email address (optional)")
        new_password        = st.text_input("Password", type="password")
        new_password_repeat = st.text_input("Confirm password", type="password")
        activation_code     = st.text_input(
            "🔑 License key",
            placeholder="Paste your license key here",
            help="You received this via email after purchase."
        )

        submit_button = st.form_submit_button(
            "✅ Activate & Create Account",
            use_container_width=True
        )

        if submit_button:
            if not all([new_username, new_password, activation_code]):
                st.error("⚠️ Username, password, and license key are required.")
            elif new_password != new_password_repeat:
                st.error("⚠️ Passwords do not match.")
            else:
                success, message = db_utils.register_new_user(
                    new_username, new_email, new_name, new_password, activation_code
                )
                if success:
                    st.success(message)
                    st.balloons()
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(message)

    st.markdown(
        "<div style='color:#888; font-size:13px; margin-top:8px;'>"
        "Already have an account? Click the <strong>Login</strong> tab above.<br>"
        "Need to purchase a license? Click the <strong>Purchase</strong> tab above."
        "</div>",
        unsafe_allow_html=True
    )

# ──────────────────────────────────────────────────────────
# Tab 5: Login
# ──────────────────────────────────────────────────────────
with tab_login:
    try:
        authenticator.login('Login', 'main')
    except TypeError:
        authenticator.login(location='main')
    authentication_status = st.session_state.get("authentication_status")
    name = st.session_state.get("name")
    username = st.session_state.get("username")

    if authentication_status:
        db_token = db_utils.get_session_token(username)

        if "session_token" not in st.session_state:
            new_token = str(uuid.uuid4())
            st.session_state["session_token"] = new_token
            db_utils.update_session_token(username, new_token)
        elif st.session_state["session_token"] != db_token:
            st.error("🚨 Your account was signed in from another device. You have been signed out for security.")
            if st.button("Sign back in here (will sign out other devices)"):
                new_token = str(uuid.uuid4())
                st.session_state["session_token"] = new_token
                db_utils.update_session_token(username, new_token)
                st.rerun()
            st.stop()

        st.success(f'⚡ Welcome, {name}!')

        if st.button("Sign out"):
            db_utils.update_session_token(username, "")
            if "session_token" in st.session_state:
                del st.session_state["session_token"]
            st.session_state["authentication_status"] = None
            st.session_state["name"] = None
            st.session_state["username"] = None
            st.rerun()

        st.info("👈 You're in! Click **LCL Filter Design** in the left sidebar to start.")

        st.markdown("---")
        st.markdown("### 📢 Latest Updates")
        st.markdown("""
        - **v1.0** — All 10 modules live. Early access is open.
        - Feedback welcome: use the **Feedback** page in the sidebar.
        """)

    elif authentication_status == False:
        st.error("❌ Incorrect username or password.")
    elif authentication_status is None:
        st.warning("Enter your username and password to log in.")

# ── Footer ────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 12px; padding: 10px 0;'>
        © 2025 PowerTool · Built by a power electronics engineer, for power electronics engineers<br>
        Contact: <a href='mailto:gpy1090@gmail.com' style='color: gray;'>gpy1090@gmail.com</a>
    </div>
    """,
    unsafe_allow_html=True
)
