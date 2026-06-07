# PowerTool EN — Deployment Guide

## Files in this package
- `Home.py` — Main landing page (login, purchase, demo)
- `pages/01_LCL_Filter_Design.py` — Core LCL filter design tool (10,000+ lines)
- `pages/02_Feedback.py` — User feedback page
- `utils/` — Algorithm library (lcl_algo, inverter_algo, vector_algo)
- `db_utils.py` — User database and authentication
- `requirements.txt` — Python dependencies

## Deployment Steps

### 1. Push to GitHub
Create a new GitHub repository and push all files in this folder.

### 2. Deploy on Streamlit Community Cloud (Free)
1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Click "New app" → select your repo
4. Set Main file path: `Home.py`
5. Click Deploy

Your app will be live at: `https://your-app-name.streamlit.app`

### 3. Payment Integration (Stripe)
- Sign up at https://stripe.com
- Create a Payment Link for $9.99 one-time purchase
- Replace `https://buy.stripe.com/YOUR_LINK_HERE` in `Home.py`
- Stripe sends payment confirmation to your email
- Manually issue license keys via `gen_keys.py` initially

### 4. First Marketing Post (Reddit)
Post to r/PowerElectronics:

Title: "I built a free LCL filter design tool for grid-connected inverters — covers full workflow from parameter calculation to capacitor selection"

Body: Describe your background as a wind power converter engineer, the problem you solved, and link to the tool. Start with free access to build users.

## TODO before going live
- [ ] Replace Stripe payment link in Home.py
- [ ] Replace support email in Home.py  
- [ ] Record a new English demo video (or add subtitles)
- [ ] Update the Bilibili video link or replace with YouTube
- [ ] Test full registration flow on Streamlit Cloud
