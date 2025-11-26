/* Lightweight shared navigation helpers for PubTrends pages
 * Responsibilities:
 *  - Desktop scroll-spy: highlight current section link inside a sidebar nav
 *  - Optional: update a label with the current section name
 *  - Back-to-top button visibility + smooth scroll
 *
 * Usage (per page):
 *  window.PTNavigation.init({
 *    desktopNavId: 'onpage-desktop',
 *    currentLabelId: 'current-card-name', // optional
 *    backToTopId: 'backToTop',            // optional
 *    rootMargin: '-35% 0px -50% 0px'      // optional
 *  });
 */
(function (global) {
  'use strict';

  function init(options) {
    options = options || {};
    var desktopNavId = options.desktopNavId || 'onpage-desktop';
    var currentLabelId = options.currentLabelId || null;
    var backToTopId = options.backToTopId || null;
    var rootMargin = options.rootMargin || '-35% 0px -50% 0px';

    // Back-to-top toggle + click handler (optional)
    if (backToTopId) {
      var backBtn = document.getElementById(backToTopId);
      if (backBtn) {
        function toggleBackBtn() {
          if (window.scrollY > 400) backBtn.classList.add('show');
          else backBtn.classList.remove('show');
        }
        toggleBackBtn();
        window.addEventListener('scroll', toggleBackBtn, { passive: true });
        backBtn.addEventListener('click', function () {
          window.scrollTo({ top: 0, behavior: 'smooth' });
        });
      }
    }

    // Desktop scroll-spy (optional if nav not present)
    var desktopNav = document.getElementById(desktopNavId);
    if (!desktopNav) return;

    // Map section id -> {linkEl, title}
    var linkMap = {};
    var links = desktopNav.querySelectorAll('a.nav-link[href^="#"]');
    links.forEach(function (a) {
      var hash = a.getAttribute('href');
      if (!hash) return;
      var id = hash.slice(1);
      linkMap[id] = { linkEl: a, title: (a.textContent || '').trim() };
    });

    var currentNameEl = currentLabelId ? document.getElementById(currentLabelId) : null;

    function setActive(id) {
      if (!id || !linkMap[id]) return;
      Object.keys(linkMap).forEach(function (k) {
        linkMap[k].linkEl.classList.toggle('active', k === id);
      });
      if (currentNameEl) currentNameEl.textContent = linkMap[id].title;
    }

    var sections = Object.keys(linkMap)
      .map(function (id) { return document.getElementById(id); })
      .filter(function (el) { return !!el; });
    if (sections.length === 0) return;

    var visible = {};
    var rafId = null;
    function updateActiveFromVisible() {
      rafId = null;
      var bestId = null;
      var bestRatio = 0;
      Object.keys(visible).forEach(function (id) {
        var ratio = visible[id] || 0;
        if (ratio > bestRatio) { bestRatio = ratio; bestId = id; }
      });
      if (bestId) setActive(bestId);
    }

    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        var id = entry.target.id;
        visible[id] = entry.isIntersecting ? entry.intersectionRatio : 0;
      });
      if (!rafId) rafId = requestAnimationFrame(updateActiveFromVisible);
    }, {
      root: null,
      rootMargin: rootMargin,
      threshold: [0, 0.01, 0.1, 0.25, 0.5, 0.75, 1]
    });

    sections.forEach(function (sec) { observer.observe(sec); });

    // Initialize selection based on current scroll position
    setTimeout(function () {
      var bestId = null;
      var bestDist = Infinity;
      sections.forEach(function (sec) {
        var rect = sec.getBoundingClientRect();
        var dist = Math.abs(rect.top - (window.innerHeight * 0.3));
        if (dist < bestDist) { bestDist = dist; bestId = sec.id; }
      });
      if (bestId) setActive(bestId);
    }, 0);
  }

  global.PTNavigation = {
    init: init
  };
})(window);
