// Firebase sign-in for the AuraFix Streamlit component.
//
// Loaded as an ES module (type="module") so it can import the Firebase Web
// SDK from gstatic and so no inline <script> is needed — which keeps the
// app's Content-Security-Policy strict (script-src 'self' + gstatic only).
//
// It speaks Streamlit's component postMessage protocol directly (no build
// step): announces readiness, receives the Firebase web config via the
// render event, and — after a user completes a popup sign-in — posts the
// resulting ID token back to Python, which verifies it server-side.

import { initializeApp, getApps } from "https://www.gstatic.com/firebasejs/10.12.0/firebase-app.js";
import {
  getAuth,
  setPersistence,
  inMemoryPersistence,
  GoogleAuthProvider,
  GithubAuthProvider,
  signInWithPopup,
} from "https://www.gstatic.com/firebasejs/10.12.0/firebase-auth.js";

const $ = (id) => document.getElementById(id);
const msg = $("msg");

let auth = null;
let initialised = false;
let busy = false;

// --- Streamlit protocol helpers -------------------------------------------
function post(type, extra) {
  window.parent.postMessage({ isStreamlitMessage: true, type, ...extra }, "*");
}
function setValue(value) {
  post("streamlit:setComponentValue", { value, dataType: "json" });
}
function sendHeight() {
  const h = document.documentElement.scrollHeight;
  post("streamlit:setFrameHeight", { height: h });
}
function ready() {
  post("streamlit:componentReady", { apiVersion: 1 });
}

function showMessage(text, isError) {
  msg.textContent = text || "";
  msg.className = isError ? "msg error" : "msg";
  sendHeight();
}

function setBusy(state) {
  busy = state;
  document.querySelectorAll("button.sso").forEach((b) => (b.disabled = state));
}

// --- Firebase --------------------------------------------------------------
function ensureFirebase(config) {
  if (initialised) return true;
  if (!config || !config.apiKey || !config.authDomain || !config.projectId) {
    showMessage("Sign-in is not configured yet.", true);
    return false;
  }
  try {
    const app = getApps().length ? getApps()[0] : initializeApp(config);
    auth = getAuth(app);
    // Keep nothing in browser storage: the token only needs to live long
    // enough to hand to the server, which then owns the session.
    setPersistence(auth, inMemoryPersistence).catch(() => {});
    initialised = true;
    return true;
  } catch (e) {
    showMessage("Could not initialise sign-in.", true);
    return false;
  }
}

async function signIn(kind) {
  if (busy || !auth) return;
  setBusy(true);
  showMessage("Opening sign-in window…", false);
  try {
    let provider;
    if (kind === "google") {
      provider = new GoogleAuthProvider();
    } else if (kind === "github") {
      provider = new GithubAuthProvider();
    } else {
      showMessage("That sign-in method isn't available.", true);
      setBusy(false);
      return;
    }
    const result = await signInWithPopup(auth, provider);
    const user = result.user;
    const idToken = await user.getIdToken(/* forceRefresh */ true);
    const profile = {
      email: user.email || "",
      name: user.displayName || "",
      picture: user.photoURL || "",
    };
    showMessage("Signed in. Loading…", false);
    // A fresh nonce so Python always sees a new value and can ignore one it
    // has already consumed.
    setValue({ idToken, profile, nonce: cryptoNonce() });
  } catch (e) {
    const code = (e && e.code) || "";
    if (code === "auth/popup-closed-by-user" || code === "auth/cancelled-popup-request") {
      showMessage("Sign-in cancelled.", false);
    } else if (code === "auth/account-exists-with-different-credential") {
      showMessage("That email is already registered with a different provider.", true);
    } else {
      showMessage("Sign-in failed. Please try again.", true);
    }
    setBusy(false);
  }
}

function cryptoNonce() {
  try {
    return crypto.randomUUID();
  } catch (_) {
    return String(Date.now()) + Math.random().toString(16).slice(2);
  }
}

// --- Render ----------------------------------------------------------------
function onRender(event) {
  const data = event.data;
  if (!data || data.type !== "streamlit:render") return;
  const args = data.args || {};
  const config = args.config || {};
  const providers = Array.isArray(args.providers) ? args.providers : ["google"];

  if (!ensureFirebase(config)) {
    sendHeight();
    return;
  }

  const wire = (id, kind) => {
    const btn = $(id);
    if (providers.includes(kind)) {
      btn.hidden = false;
      if (!btn._wired) {
        btn.addEventListener("click", () => signIn(kind));
        btn._wired = true;
      }
    } else {
      btn.hidden = true;
    }
  };
  wire("btn-google", "google");
  wire("btn-github", "github");
  // Email sign-in is not wired in this minimal component; google/github cover
  // the requested flows. The button stays hidden unless explicitly enabled.
  $("btn-email").hidden = true;

  if (!msg.textContent) showMessage("", false);
  sendHeight();
}

window.addEventListener("message", onRender);
window.addEventListener("resize", sendHeight);
if (window.ResizeObserver) {
  new ResizeObserver(sendHeight).observe(document.documentElement);
}

ready();
sendHeight();
