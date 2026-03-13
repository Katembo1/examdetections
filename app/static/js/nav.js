/* nav.js — tab switching + mobile drawer (part of base layout) */
function switchTab(tabName) {
  document.querySelectorAll('.nav-tab, .drawer-tab').forEach((t) => {
    t.classList.toggle('active', t.dataset.tab === tabName);
  });
  document.querySelectorAll('.panel').forEach((p) => p.classList.remove('active'));
  const targetPanel = document.getElementById('tab-' + tabName);
  if (targetPanel) targetPanel.classList.add('active');
}

document.querySelectorAll('.nav-tab').forEach((tab) => {
  tab.addEventListener('click', () => switchTab(tab.dataset.tab));
});

const hamburger     = document.getElementById('hamburger');
const drawer        = document.getElementById('drawer');
const drawerOverlay = document.getElementById('drawerOverlay');
const drawerClose   = document.getElementById('drawerClose');

function openDrawer() {
  drawer.classList.add('open');
  drawerOverlay.style.display = 'block';
  requestAnimationFrame(() => drawerOverlay.classList.add('open'));
  hamburger.classList.add('open');
  hamburger.setAttribute('aria-expanded', 'true');
  document.body.style.overflow = 'hidden';
}

function closeDrawer() {
  drawer.classList.remove('open');
  drawerOverlay.classList.remove('open');
  hamburger.classList.remove('open');
  hamburger.setAttribute('aria-expanded', 'false');
  document.body.style.overflow = '';
  setTimeout(() => { drawerOverlay.style.display = ''; }, 260);
}

hamburger.addEventListener('click', openDrawer);
drawerClose.addEventListener('click', closeDrawer);
drawerOverlay.addEventListener('click', closeDrawer);

document.querySelectorAll('.drawer-tab').forEach((tab) => {
  tab.addEventListener('click', () => { switchTab(tab.dataset.tab); closeDrawer(); });
});

const THEME_STORAGE_KEY = 'examguard-theme';

function getStoredTheme() {
  try {
    const value = localStorage.getItem(THEME_STORAGE_KEY);
    if (value === 'light' || value === 'dark') return value;
  } catch (e) {
    return 'dark';
  }
  return 'dark';
}

function setStoredTheme(theme) {
  try {
    localStorage.setItem(THEME_STORAGE_KEY, theme);
  } catch (e) {
    // Ignore storage failures in private-restricted contexts.
  }
}

function applyTheme(theme) {
  const normalized = theme === 'light' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', normalized);
  setStoredTheme(normalized);
  syncThemeSettingsUi(normalized);
}

function syncThemeSettingsUi(theme) {
  const current = document.getElementById('themeCurrent');
  const lightBtn = document.getElementById('themeLightBtn');
  const darkBtn = document.getElementById('themeDarkBtn');
  if (current) current.textContent = theme === 'light' ? 'Light' : 'Dark';

  [lightBtn, darkBtn].forEach((btn) => {
    if (!btn) return;
    const selected = btn.dataset.theme === theme;
    btn.classList.toggle('is-selected', selected);
    btn.setAttribute('aria-checked', selected ? 'true' : 'false');
  });
}

function initThemeSettings() {
  const lightBtn = document.getElementById('themeLightBtn');
  const darkBtn = document.getElementById('themeDarkBtn');
  const themeButtons = [lightBtn, darkBtn].filter(Boolean);

  const initialTheme = document.documentElement.getAttribute('data-theme') || getStoredTheme();
  applyTheme(initialTheme);

  themeButtons.forEach((btn) => {
    btn.addEventListener('click', () => applyTheme(btn.dataset.theme));
  });
}

initThemeSettings();
