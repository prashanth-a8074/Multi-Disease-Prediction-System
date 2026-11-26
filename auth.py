import streamlit as st
from src.database import authenticate_user, register_user

def init_auth_session():
    """Initializes authentication variables in Streamlit session state."""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user_info' not in st.session_state:
        st.session_state.user_info = None

def render_login_signup():
    """Renders the login and registration panel on the main page."""
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="color: #4A90E2; font-family: 'Outfit', sans-serif; font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem;">
            🛡️ Multi-Disease Prediction System
        </h1>
        <p style="color: #B0C4DE; font-size: 1.1rem;">
            Secure Clinically-Validated Multi-Disease Diagnostics Dashboard
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🔐 Sign In", "📝 Create Account"])
    
    with tab1:
        st.subheader("Login to Your Account")
        login_username = st.text_input("Username", key="login_username_input")
        login_password = st.text_input("Password", type="password", key="login_password_input")
        
        if st.button("Sign In", type="primary", use_container_width=True):
            if not login_username or not login_password:
                st.error("Please fill in all fields.")
            else:
                user = authenticate_user(login_username, login_password)
                if user:
                    st.session_state.logged_in = True
                    st.session_state.user_info = user
                    st.success(f"Welcome back, {user['username']}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
                    
    with tab2:
        st.subheader("Register a New Profile")
        reg_username = st.text_input("Choose Username", key="reg_username_input")
        reg_email = st.text_input("Email Address", key="reg_email_input")
        reg_password = st.text_input("Password", type="password", key="reg_password_input")
        reg_confirm = st.text_input("Confirm Password", type="password", key="reg_confirm_input")
        
        if st.button("Sign Up", use_container_width=True):
            if not reg_username or not reg_email or not reg_password or not reg_confirm:
                st.error("All fields are required.")
            elif reg_password != reg_confirm:
                st.error("Passwords do not match.")
            elif len(reg_password) < 6:
                st.error("Password must be at least 6 characters.")
            else:
                # Attempt to register
                success = register_user(reg_username, reg_email, reg_password)
                if success:
                    st.success("Registration successful! You can now sign in.")
                else:
                    st.error("Username or email already exists.")

def logout():
    """Logs out the user and clears session state."""
    st.session_state.logged_in = False
    st.session_state.user_info = None
    st.success("Successfully logged out.")
    st.rerun()
