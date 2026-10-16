// Theme management for ClaimWise

const ThemeManager = {
    // Icons for the toggle button
    icons: {
        dark: '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>',
        light: '<circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>'
    },

    init() {
        this.applyTheme(this.getPreferredTheme());
        
        // Wait for DOM to be ready before attaching event listeners
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.attachListeners());
        } else {
            this.attachListeners();
        }
    },

    getPreferredTheme() {
        // 1. Check user override in localStorage
        const saved = localStorage.getItem('claimwise_theme');
        if (saved === 'dark' || saved === 'light') {
            return saved;
        }
        
        // 2. Default to Sage Light (as per requirements) unless system is explicitly dark
        // The requirements say "Sage Light is default". We could check system preference here:
        // if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) return 'dark';
        
        return 'light'; // Explicitly default to Light
    },

    applyTheme(theme) {
        if (theme === 'dark') {
            document.documentElement.setAttribute('data-theme', 'dark');
        } else {
            document.documentElement.removeAttribute('data-theme');
        }
        this.updateToggleButton(theme);
    },

    toggleTheme() {
        const currentTheme = document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        
        localStorage.setItem('claimwise_theme', newTheme);
        this.applyTheme(newTheme);
    },

    updateToggleButton(theme) {
        const icon = document.getElementById('themeIcon');
        const btn = document.getElementById('themeToggle');
        
        if (icon) {
            icon.innerHTML = theme === 'dark' ? this.icons.light : this.icons.dark;
        }
        
        if (btn) {
            btn.setAttribute('aria-label', `Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`);
        }
    },

    attachListeners() {
        const btn = document.getElementById('themeToggle');
        if (btn) {
            btn.addEventListener('click', () => this.toggleTheme());
        }
        
        // Optional: listen for system theme changes if we want to sync
        /*
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
            if (!localStorage.getItem('claimwise_theme')) {
                this.applyTheme(e.matches ? 'dark' : 'light');
            }
        });
        */
    }
};

// Initialize immediately to prevent flash
ThemeManager.init();
