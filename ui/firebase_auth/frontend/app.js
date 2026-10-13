// Firebase sign-in for the AuraFix Streamlit component.
//
// Loaded as an ES module (type="module") so it can import the Firebase Web
// SDK from gstatic and so no inline <script> is needed — which keeps the
// app's Content-Security-Policy strict (script-src 'self' + gstatic only).
//
// Responsibilities:
//   * Show Google / GitHub buttons and complete a popup sign-in.
//   * "Keep me signed in": when enabled, sign in with Firebase LOCAL
//     persistence so the session survives a browser restart; otherwise keep
//     it in memory only (ends with the tab). The server caps a persisted
//     session at 30 days.
//   * Auto-restore: on load, if a persisted user already exists, silently
//     hand a fresh ID token to Python so the session resumes without a click.
//   * Sign out: clear the Firebase session on request.
// It verifies nothing itself — every token is checked server-side.

import { initializeApp, getApps } from "https://www.gstatic.com/firebasejs/10.12.0/firebase-app.js";
import {
  getAuth,
  setPersistence,
  browserLocalPersistence,
  inMemoryPersistence,
  onAuthStateChanged,
  signOut,
  GoogleAuthProvider,
  GithubAuthProvider,
  signInWithPopup,
} from "https://www.gstatic.com/firebasejs/10.12.0/firebase-auth.js";

const $ = (id) => document.getElementById(id);
const msg = $("msg");

let auth = null;
let initialised = false;
let watching = false;
let busy = false;
let emitted = false;       // one sign-in/restore emit per load
let persist = false;       // "keep me signed in" state from args
let signoutMode = false;
let signedOutDone = false;

// --- Streamlit protocol helpers -------------------------------------------
function post(type, extra) {
  window.parent.postMessage({ isStreamlitMessage: true, type, ...extra }, "*");
}
function setValue(value) {
  post("streamlit:setComponentValue", { value, dataType: "json" });
}
function sendHeight() {
  post("streamlit:setFrameHeight", { height: document.documentElement.scrollHeight });
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
function cryptoNonce() {
  try { return crypto.randomUUID(); }
  catch (_) { return String(Date.now()) + Math.random().toString(16).slice(2); }
}
function profileOf(user) {
  return { email: user.email || "", name: user.displayName || "", picture: user.photoURL || "" };
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
    if (!watching) {
      watching = true;
      // Fires on load with a persisted user (from a prior "keep me signed
      // in"), and after a popup sign-in. We auto-emit only for the restore
      // case; the explicit sign-in path emits on its own.
      onAuthStateChanged(auth, (user) => {
        if (signoutMode || emitted || busy) return;
        if (user) {
          emitted = true;
          user.getIdToken(true).then((idToken) => {
            setValue({ idToken, profile: profileOf(user), nonce: cryptoNonce(), restored: true });
          }).catch(() => { emitted = false; });
        }
      });
    }
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
  msg.className = "msg";
  msg.innerHTML = '<span class="spin"></span>Opening sign-in window…';
  sendHeight();
  try {
    // Persistence must be set before the popup. Local = survives restart;
    // in-memory = gone when the tab closes.
    await setPersistence(auth, persist ? browserLocalPersistence : inMemoryPersistence);

    const provider = kind === "google" ? new GoogleAuthProvider()
                   : kind === "github" ? new GithubAuthProvider() : null;
    if (!provider) { showMessage("That sign-in method isn't available.", true); setBusy(false); return; }

    const result = await signInWithPopup(auth, provider);
    const user = result.user;
    const idToken = await user.getIdToken(true);
    emitted = true;
    showMessage("Signed in. Loading…", false);
    setValue({ idToken, profile: profileOf(user), nonce: cryptoNonce(), restored: false });
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

async function doSignOut() {
  if (signedOutDone) return;
  signedOutDone = true;
  try { if (auth) await signOut(auth); } catch (_) {}
  setValue({ signedOut: true, nonce: cryptoNonce() });
}

// --- Render ----------------------------------------------------------------
function onRender(event) {
  const data = event.data;
  if (!data || data.type !== "streamlit:render") return;
  const args = data.args || {};
  const config = args.config || {};
  const providers = Array.isArray(args.providers) ? args.providers : ["google"];
  persist = !!args.persist;

  if (!ensureFirebase(config)) { sendHeight(); return; }

  // Sign-out request: clear Firebase and report back; don't show buttons.
  if (args.signout) {
    signoutMode = true;
    document.querySelectorAll("button.sso").forEach((b) => (b.hidden = true));
    showMessage("Signing out…", false);
    doSignOut();
    return;
  }

  const wire = (id, kind) => {
    const btn = $(id);
    if (!btn) return;
    if (providers.includes(kind)) {
      btn.hidden = false;
      if (!btn._wired) { btn.addEventListener("click", () => signIn(kind)); btn._wired = true; }
    } else {
      btn.hidden = true;
    }
  };
  wire("btn-google", "google");
  wire("btn-github", "github");

  if (!msg.textContent) showMessage("", false);
  sendHeight();
}

window.addEventListener("message", onRender);
window.addEventListener("resize", sendHeight);
if (window.ResizeObserver) new ResizeObserver(sendHeight).observe(document.documentElement);

ready();
sendHeight();
