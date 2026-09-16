/**
 * Data-Juicer Sphinx Theme - JavaScript
 */

(function() {
  'use strict';

  function findHashTarget(hash) {
    if (!hash || hash === '#') return null;
    var id = hash.slice(1);
    try { id = decodeURIComponent(id); } catch (err) { /* Keep literal malformed fragments. */ }
    return document.getElementById(id);
  }

  // ==================== Dark Mode ====================
  function syncPygmentsDark(theme) {
    var link = document.getElementById('pygments-dark-css');
    if (link) link.disabled = (theme !== 'dark');
  }

  function initTheme() {
    const stored = localStorage.getItem('dj-theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const theme = stored || (prefersDark ? 'dark' : 'light');
    document.documentElement.setAttribute('data-theme', theme);
    syncPygmentsDark(theme);
  }

  function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('dj-theme', next);
    syncPygmentsDark(next);
  }

  // ==================== Sidebar ====================
  function initSidebar() {
    const toggle = document.getElementById('sidebar-toggle');
    const sidebar = document.getElementById('sidebar');
    const overlay = document.getElementById('sidebar-overlay');

    if (!toggle || !sidebar) return;

    toggle.addEventListener('click', function() {
      sidebar.classList.toggle('open');
      overlay.classList.toggle('visible');
    });

    if (overlay) {
      overlay.addEventListener('click', function() {
        sidebar.classList.remove('open');
        overlay.classList.remove('visible');
      });
    }
  }

  // ==================== Dropdowns ====================
  function initDropdowns() {
    document.querySelectorAll('.dropdown').forEach(function(dropdown) {
      var trigger = dropdown.querySelector('.dropdown-trigger');
      if (!trigger) return;

      trigger.addEventListener('click', function(e) {
        e.stopPropagation();
        var wasActive = dropdown.classList.contains('active');
        closeAllDropdowns();
        if (!wasActive) {
          dropdown.classList.add('active');
        }
      });
    });

    document.addEventListener('click', closeAllDropdowns);
  }

  function closeAllDropdowns() {
    document.querySelectorAll('.dropdown.active').forEach(function(d) {
      d.classList.remove('active');
    });
  }

  // ==================== Copy Buttons ====================
  function initCopyButtons() {
    document.querySelectorAll('.highlight pre, .article > pre').forEach(function(pre) {
      if (pre.querySelector('.copy-button')) return;

      var wrapper = pre.closest('.highlight') || pre;
      wrapper.style.position = 'relative';

      var btn = document.createElement('button');
      btn.className = 'copy-button';
      btn.setAttribute('aria-label', 'Copy code');
      btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="5" y="5" width="9" height="9" rx="1.5"/><path d="M3 11V2.5A1.5 1.5 0 014.5 1H11"/></svg>';

      btn.addEventListener('click', function() {
        var code = pre.textContent || pre.innerText;
        navigator.clipboard.writeText(code).then(function() {
          btn.classList.add('copied');
          btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3.5 8.5 6.5 11.5 12.5 5.5"/></svg>';
          setTimeout(function() {
            btn.classList.remove('copied');
            btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="5" y="5" width="9" height="9" rx="1.5"/><path d="M3 11V2.5A1.5 1.5 0 014.5 1H11"/></svg>';
          }, 2000);
        });
      });

      wrapper.appendChild(btn);
    });
  }

  // ==================== TOC Active Tracking ====================
  var tocScrollHandler = null;

  function initTocTracking() {
    if (tocScrollHandler) {
      window.removeEventListener('scroll', tocScrollHandler);
      tocScrollHandler = null;
    }

    var tocLinks = document.querySelectorAll('.toc-sidebar a');
    if (!tocLinks.length) return;

    var headings = [];
    tocLinks.forEach(function(link) {
      var id = link.getAttribute('href');
      if (id && id.startsWith('#')) {
        var heading = findHashTarget(id);
        if (heading) headings.push({ el: heading, link: link });
      }
    });

    if (!headings.length) return;

    function updateActive() {
      var scrollTop = window.scrollY + 100;
      var active = headings[0];

      for (var i = 0; i < headings.length; i++) {
        if (headings[i].el.offsetTop <= scrollTop) {
          active = headings[i];
        }
      }

      tocLinks.forEach(function(l) { l.classList.remove('active'); });
      if (active) active.link.classList.add('active');
    }

    tocScrollHandler = updateActive;
    window.addEventListener('scroll', updateActive, { passive: true });
    updateActive();
  }

  // ==================== Sidebar Collapsible ====================
  function getSidebarKey(li) {
    var link = li.querySelector(':scope > a');
    if (!link) return null;
    var href = link.getAttribute('href');
    if (href && href !== '#') return link.pathname;
    return link.textContent.trim();
  }

  function saveSidebarState(nav) {
    var expanded = [];
    nav.querySelectorAll('li.has-children.expanded').forEach(function(li) {
      var key = getSidebarKey(li);
      if (key) expanded.push(key);
    });
    try { sessionStorage.setItem('dj-sidebar-expanded', JSON.stringify(expanded)); } catch(e) {}
  }

  function restoreSidebarState(nav) {
    var stored = null;
    try { stored = sessionStorage.getItem('dj-sidebar-expanded'); } catch(e) {}
    if (!stored) return;
    var expanded = JSON.parse(stored);
    if (!Array.isArray(expanded)) return;
    nav.querySelectorAll('li.has-children').forEach(function(li) {
      var key = getSidebarKey(li);
      if (key && expanded.indexOf(key) !== -1) {
        li.classList.add('expanded');
      }
    });
  }

  function initSidebarCollapse() {
    var nav = document.querySelector('.sidebar-nav');
    if (!nav) return;

    // Mark items that have children and add toggle chevrons
    nav.querySelectorAll('li').forEach(function(li) {
      var childUl = li.querySelector(':scope > ul');
      if (childUl && childUl.querySelector('li')) {
        li.classList.add('has-children');
        var link = li.querySelector(':scope > a');
        if (link && !link.querySelector('.toc-toggle')) {
          var toggle = document.createElement('span');
          toggle.className = 'toc-toggle';
          toggle.innerHTML = '<svg width="12" height="12" viewBox="0 0 12 12" fill="none"><path d="M4.5 2.5l4 3.5-4 3.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>';
          link.style.position = 'relative';
          link.appendChild(toggle);
        }
      }
    });

    // Restore user-toggled expanded sections
    restoreSidebarState(nav);

    // Find the current page item:
    // 1. Use Sphinx-generated "current" class on <a> (most reliable)
    // 2. Fallback: match resolved pathname
    var currentLi = null;
    var currentLink = nav.querySelector('a.current');
    if (currentLink) {
      currentLi = currentLink.closest('li');
    }

    if (!currentLi) {
      var currentPath = window.location.pathname;
      var links = nav.querySelectorAll('a');
      for (var i = 0; i < links.length; i++) {
        var href = links[i].getAttribute('href');
        if (!href || href === '#') continue;
        if (links[i].pathname === currentPath) {
          currentLi = links[i].closest('li');
          break;
        }
      }
    }

    // Also check the sidebar-home entry
    if (!currentLi) {
      var homeCurrent = nav.querySelector('.sidebar-home li.current');
      if (homeCurrent) currentLi = homeCurrent;
    }

    // Mark current and expand all ancestors
    if (currentLi) {
      currentLi.classList.add('current');
      var parent = currentLi.parentElement;
      while (parent && parent !== nav) {
        if (parent.tagName === 'LI') {
          parent.classList.add('expanded');
        }
        parent = parent.parentElement;
      }
    }

    // Toggle on click — persist state
    nav.addEventListener('click', function(e) {
      var toggle = e.target.closest('.toc-toggle');
      if (toggle) {
        e.preventDefault();
        e.stopPropagation();
        var li = toggle.closest('li');
        if (li) {
          li.classList.toggle('expanded');
          saveSidebarState(nav);
        }
      }
    });
  }

  // ==================== Sidebar Active State ====================
  function initSidebarActive() {
    initSidebarCollapse();
    // Scroll active item into view
    var currentItem = document.querySelector('.sidebar-nav li.current > a');
    if (currentItem) {
      var sidebar = document.getElementById('sidebar');
      if (sidebar) {
        currentItem.scrollIntoView({ block: 'nearest' });
      }
    }
  }

  // ==================== Search Modal ====================
  function initSearch() {
    var trigger = document.getElementById('search-trigger');
    var overlay = document.getElementById('search-overlay');
    var input = document.getElementById('search-modal-input');

    if (!trigger || !overlay) return;

    function openSearch() {
      overlay.classList.add('visible');
      if (input) input.focus();
    }

    function closeSearch() {
      overlay.classList.remove('visible');
      if (input) input.value = '';
    }

    trigger.addEventListener('click', openSearch);

    overlay.addEventListener('click', function(e) {
      if (e.target === overlay) closeSearch();
    });

    document.addEventListener('keydown', function(e) {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        if (overlay.classList.contains('visible')) {
          closeSearch();
        } else {
          openSearch();
        }
      }
      if (e.key === 'Escape' && overlay.classList.contains('visible')) {
        closeSearch();
      }
    });

    if (input) {
      input.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && input.value.trim()) {
          var searchUrl = document.querySelector('link[rel="search"]');
          if (searchUrl) {
            window.location.href = searchUrl.href.replace('search.html', '') + 'search.html?q=' + encodeURIComponent(input.value);
          } else {
            window.location.href = 'search.html?q=' + encodeURIComponent(input.value);
          }
        }
      });
    }
  }

  // ==================== Version switcher ====================
  function initVersionSwitcher() {
    var dropdown = document.getElementById('version-dropdown');
    if (!dropdown) return;
    var url = dropdown.getAttribute('data-versions-url');
    if (!url) return;
    var prefix = dropdown.getAttribute('data-link-prefix') || '../';
    var page = dropdown.getAttribute('data-page') || 'index';
    var current = dropdown.getAttribute('data-current') || 'main';

    function render(versions) {
      var panel = dropdown.querySelector('.dropdown-panel');
      if (!panel || !versions || !versions.length) return;
      panel.innerHTML = versions.map(function(v) {
        var href = prefix + v + '/' + page + '.html';
        var cls = 'dropdown-item' + (v === current ? ' active' : '');
        return '<a href="' + href + '" class="' + cls + '">' + v + '</a>';
      }).join('');
    }

    var cacheKey = 'dj-versions:' + new URL(url, window.location.href).href;
    var cached = null;
    try {
      cached = sessionStorage.getItem(cacheKey);
    } catch (e) { /* sessionStorage unavailable */ }
    if (cached) {
      try { render(JSON.parse(cached)); return; } catch (e) { /* refetch below */ }
    }

    fetch(url).then(function(resp) {
      if (!resp.ok) throw new Error('HTTP ' + resp.status);
      return resp.json();
    }).then(function(data) {
      if (data && Array.isArray(data.versions) && data.versions.length) {
        try { sessionStorage.setItem(cacheKey, JSON.stringify(data.versions)); } catch (e) {}
        render(data.versions);
      }
    }).catch(function() {});
  }

  // ==================== SPA Navigation ====================
  var spaController = null;

  function shouldInterceptLink(anchor, event) {
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return false;
    if (event.button !== 0) return false;
    if (anchor.hasAttribute('data-spa-bypass')) return false;
    if (anchor.hasAttribute('download')) return false;

    var target = anchor.getAttribute('target');
    if (target && target !== '_self') return false;

    if (anchor.origin !== window.location.origin) return false;

    var pathname = anchor.pathname;
    if (pathname.includes('search.html')) return false;

    var lastSegment = pathname.split('/').pop();
    if (lastSegment && lastSegment.includes('.')) {
      var ext = lastSegment.split('.').pop().toLowerCase();
      if (ext !== 'html' && ext !== 'htm') return false;
    }

    if (anchor.closest('#version-dropdown, #lang-dropdown')) return false;
    if (anchor.closest('.ask-ai-widget')) return false;

    return true;
  }

  function showLoadingBar() {
    var bar = document.getElementById('spa-loading-bar');
    if (!bar) {
      bar = document.createElement('div');
      bar.id = 'spa-loading-bar';
      document.body.appendChild(bar);
    }
    bar.classList.remove('complete');
    bar.offsetWidth; // force reflow
    bar.classList.add('active');
  }

  function hideLoadingBar() {
    var bar = document.getElementById('spa-loading-bar');
    if (bar) {
      bar.classList.add('complete');
      setTimeout(function() {
        bar.classList.remove('active', 'complete');
      }, 400);
    }
  }

  function spaNavigateTo(url, pushState) {
    if (spaController) {
      spaController.abort();
    }

    var controller = new AbortController();
    spaController = controller;

    showLoadingBar();

    fetch(url, { signal: controller.signal })
      .then(function(resp) {
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        return resp.text();
      })
      .then(function(html) {
        if (controller.signal.aborted) return;

        var parser = new DOMParser();
        var newDoc = parser.parseFromString(html, 'text/html');

        var newMain = newDoc.querySelector('main.main-content');
        var newSidebar = newDoc.querySelector('aside.sidebar#sidebar');
        var newLangDropdown = newDoc.querySelector('#lang-dropdown .dropdown-panel');
        var currentMain = document.querySelector('main.main-content');
        var currentSidebar = document.querySelector('aside.sidebar#sidebar');

        if (!newMain || !currentMain) {
          window.location.href = url;
          return;
        }

        // Preserve AI-panel-related body classes
        var aiPanelOpen = document.body.classList.contains('ai-panel-open');
        var aiPanelResizing = document.body.classList.contains('ai-panel-resizing');

        currentMain.innerHTML = newMain.innerHTML;

        if (newSidebar && currentSidebar) {
          currentSidebar.innerHTML = newSidebar.innerHTML;
        }

        // Update language switcher links for the new page
        if (newLangDropdown) {
          var currentLangDropdown = document.querySelector('#lang-dropdown .dropdown-panel');
          if (currentLangDropdown) {
            currentLangDropdown.innerHTML = newLangDropdown.innerHTML;
          }
        }

        // Section links are relative to the current page, just like the sidebar.
        var sections = document.querySelector('.navbar-sections');
        var newSections = newDoc.querySelector('.navbar-sections');
        if (sections && newSections) sections.innerHTML = newSections.innerHTML;

        // Keep version destinations in sync after navigating to a nested page.
        var versionDropdown = document.querySelector('#version-dropdown');
        var newVersionDropdown = newDoc.querySelector('#version-dropdown');
        if (versionDropdown && newVersionDropdown) {
          ['data-versions-url', 'data-link-prefix', 'data-page', 'data-current'].forEach(function(name) {
            versionDropdown.setAttribute(name, newVersionDropdown.getAttribute(name) || '');
          });
          versionDropdown.querySelector('.dropdown-panel').innerHTML =
            newVersionDropdown.querySelector('.dropdown-panel').innerHTML;
          initVersionSwitcher();
        }

        document.title = newDoc.title;

        if (aiPanelOpen) document.body.classList.add('ai-panel-open');
        if (aiPanelResizing) document.body.classList.add('ai-panel-resizing');

        if (pushState !== false) {
          history.pushState({ scrollY: 0, url: url }, '', url);
        }

        // Re-initialize content-dependent features
        initContent();

        // Handle hash scrolling
        var hashTarget = window.location.hash;
        if (hashTarget) {
          var el = findHashTarget(hashTarget);
          if (el) {
            el.scrollIntoView({ behavior: 'smooth' });
          }
        } else {
          window.scrollTo(0, 0);
        }

        // Close mobile sidebar if open
        var sidebar = document.getElementById('sidebar');
        var overlay = document.getElementById('sidebar-overlay');
        if (sidebar) sidebar.classList.remove('open');
        if (overlay) overlay.classList.remove('visible');

        hideLoadingBar();
      })
      .catch(function(err) {
        if (err.name === 'AbortError') return;
        hideLoadingBar();
        window.location.href = url;
      })
      .finally(function() {
        if (spaController === controller) {
          spaController = null;
        }
      });
  }

  function initSPA() {
    if (!window.fetch || !window.DOMParser || !window.history.pushState) return;

    history.replaceState({ scrollY: 0, url: window.location.href }, '', window.location.href);

    document.addEventListener('click', function(e) {
      var anchor = e.target.closest('a');
      if (!anchor) return;

      if (!shouldInterceptLink(anchor, e)) return;

      // Same-page hash link
      if (anchor.pathname === window.location.pathname && anchor.hash) {
        e.preventDefault();
        var target = findHashTarget(anchor.hash);
        if (target) {
          target.scrollIntoView({ behavior: 'smooth' });
          history.pushState(null, '', anchor.hash);
        }
        return;
      }

      e.preventDefault();
      spaNavigateTo(anchor.href, true);
    });

    window.addEventListener('popstate', function(event) {
      spaNavigateTo(window.location.href, false);
    });
  }

  // ==================== Content Init (repeatable) ====================
  function initContent() {
    initCopyButtons();
    initTocTracking();
    initSidebarActive();
  }

  // ==================== Init ====================
  initTheme();

  document.addEventListener('DOMContentLoaded', function() {
    var themeToggle = document.getElementById('theme-toggle');
    if (themeToggle) {
      themeToggle.addEventListener('click', toggleTheme);
    }

    // Shell init (once)
    initSidebar();
    initDropdowns();
    initVersionSwitcher();
    initSearch();

    // Content init (repeatable)
    initContent();

    // SPA navigation
    initSPA();
  });
})();
