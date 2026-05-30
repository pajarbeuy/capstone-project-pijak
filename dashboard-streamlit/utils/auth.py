import hmac
import os

import streamlit as st


DEFAULT_USERNAME = "admin"
DEFAULT_PASSWORD = "admin123"


def _secret_value(section: str, key: str, default: str | None = None) -> str | None:
    try:
        return st.secrets.get(section, {}).get(key, default)
    except Exception:
        return default


def auth_credentials() -> tuple[str, str]:
    username = os.getenv("APP_USERNAME") or _secret_value("auth", "username", DEFAULT_USERNAME)
    password = os.getenv("APP_PASSWORD") or _secret_value("auth", "password", DEFAULT_PASSWORD)
    return str(username), str(password)


def is_authenticated() -> bool:
    return bool(st.session_state.get("authenticated", False))


def logout_button() -> None:
    if is_authenticated() and st.button("Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.pop("auth_username", None)
        st.rerun()


def require_login() -> None:
    if is_authenticated():
        return

    expected_username, expected_password = auth_credentials()

    st.markdown(
        """
        <div class="login-hero">
            <h1>Login Inference</h1>
            <p>Masuk untuk membuka dashboard dan halaman prediksi sentimen.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username", placeholder="Masukkan username")
        password = st.text_input("Password", placeholder="Masukkan password", type="password")
        submitted = st.form_submit_button("Login", type="primary", use_container_width=True)

    if submitted:
        valid_username = hmac.compare_digest(username.strip(), expected_username)
        valid_password = hmac.compare_digest(password, expected_password)
        if valid_username and valid_password:
            st.session_state.authenticated = True
            st.session_state.auth_username = username.strip()
            st.rerun()
        else:
            st.error("Username atau password salah.")

    if expected_username == DEFAULT_USERNAME and expected_password == DEFAULT_PASSWORD:
        st.warning("Login default aktif: username `admin`, password `admin123`")

    st.stop()
