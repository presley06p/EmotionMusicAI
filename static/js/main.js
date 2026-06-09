/* ═══════════════════════════════════════════════════════════════════════════
   EmoTune AI — main.js
   Shared utilities: theme toggle, nav scroll, auto-dismiss toasts
═══════════════════════════════════════════════════════════════════════════ */

// ── Theme Toggle ─────────────────────────────────────────────────────────
const THEME_KEY = 'emotune-theme';

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  const icon = document.getElementById('themeIcon');
  if (icon) {
    icon.className = theme === 'dark' ? 'bi bi-moon-stars' : 'bi bi-sun';
  }
}

function initTheme() {
  const saved = localStorage.getItem(THEME_KEY) || 'dark';
  applyTheme(saved);
}

document.addEventListener('DOMContentLoaded', () => {
  initTheme();

  const toggleBtn = document.getElementById('themeToggle');
  if (toggleBtn) {
    toggleBtn.addEventListener('click', () => {
      const current = document.documentElement.getAttribute('data-theme');
      const next = current === 'dark' ? 'light' : 'dark';
      applyTheme(next);
      localStorage.setItem(THEME_KEY, next);
    });
  }

  // Navbar scroll effect
  const nav = document.getElementById('mainNav');
  if (nav) {
    const onScroll = () => {
      nav.classList.toggle('scrolled', window.scrollY > 30);
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
  }

  // Auto-dismiss flash toasts after 4s
  document.querySelectorAll('.flash-toast').forEach(toast => {
    setTimeout(() => {
      toast.style.transition = 'opacity .4s';
      toast.style.opacity = '0';
      setTimeout(() => toast.remove(), 400);
    }, 4000);
  });
});
