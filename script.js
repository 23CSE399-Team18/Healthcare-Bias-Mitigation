/* ============================================
   INTERACTIVE JAVASCRIPT
   Healthcare Bias Mitigation Website
   ============================================ */

document.addEventListener('DOMContentLoaded', () => {

    // ---- Sidebar Toggle ----
    const sidebar = document.getElementById('sidebar');
    const sidebarToggle = document.getElementById('sidebarToggle');
    const mobileMenuBtn = document.getElementById('mobileMenuBtn');

    sidebarToggle.addEventListener('click', () => {
        sidebar.classList.toggle('collapsed');
    });

    mobileMenuBtn.addEventListener('click', () => {
        sidebar.classList.toggle('mobile-open');
        mobileMenuBtn.classList.toggle('active');
    });

    // Close mobile menu on nav click
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', () => {
            if (window.innerWidth <= 768) {
                sidebar.classList.remove('mobile-open');
                mobileMenuBtn.classList.remove('active');
            }
        });
    });

    // ---- Active Nav on Scroll ----
    const sections = document.querySelectorAll('.section');
    const navItems = document.querySelectorAll('.nav-item');

    const observerOptions = {
        root: null,
        rootMargin: '-20% 0px -60% 0px',
        threshold: 0
    };

    const sectionObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const sectionId = entry.target.id;
                navItems.forEach(item => {
                    item.classList.toggle('active', item.dataset.section === sectionId);
                });
            }
        });
    }, observerOptions);

    sections.forEach(section => sectionObserver.observe(section));

    // ---- Reveal on Scroll ----
    const revealCards = document.querySelectorAll('.reveal-card');
    
    const revealObserver = new IntersectionObserver((entries) => {
        entries.forEach((entry, index) => {
            if (entry.isIntersecting) {
                // Stagger the animation
                const delay = Array.from(entry.target.parentElement.children)
                    .filter(el => el.classList.contains('reveal-card'))
                    .indexOf(entry.target) * 100;
                
                setTimeout(() => {
                    entry.target.classList.add('visible');
                }, delay);
                
                revealObserver.unobserve(entry.target);
            }
        });
    }, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });

    revealCards.forEach(card => revealObserver.observe(card));

    // ---- Algorithm Tabs ----
    const algoTabs = document.querySelectorAll('.algo-tab');
    const algoPanels = document.querySelectorAll('.algo-panel');

    algoTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const targetTab = tab.dataset.tab;

            algoTabs.forEach(t => t.classList.remove('active'));
            algoPanels.forEach(p => p.classList.remove('active'));

            tab.classList.add('active');
            document.getElementById(`tab-${targetTab}`).classList.add('active');
        });
    });

    // ---- Methodology Accordion ----
    const accordionHeaders = document.querySelectorAll('.accordion-header');

    accordionHeaders.forEach(header => {
        header.addEventListener('click', () => {
            const content = header.nextElementSibling;
            const isActive = header.classList.contains('active');

            // Close all
            accordionHeaders.forEach(h => {
                h.classList.remove('active');
                h.nextElementSibling.classList.remove('active');
            });

            // Open clicked (if wasn't active)
            if (!isActive) {
                header.classList.add('active');
                content.classList.add('active');
            }
        });
    });

    // ---- Smooth scroll for anchor links ----
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                const offset = 20;
                const targetPos = target.getBoundingClientRect().top + window.pageYOffset - offset;
                window.scrollTo({ top: targetPos, behavior: 'smooth' });
            }
        });
    });

    // ---- Parallax effect on hero circles ----
    const heroSection = document.querySelector('.hero-section');
    if (heroSection) {
        window.addEventListener('scroll', () => {
            const scrolled = window.pageYOffset;
            const circles = heroSection.querySelectorAll('.hero-circle');
            circles.forEach((circle, i) => {
                const speed = 0.1 + (i * 0.05);
                circle.style.transform = `translate(${Math.sin(scrolled * 0.002 + i) * 20}px, ${scrolled * speed * -1}px)`;
            });
        }, { passive: true });
    }

    // ---- Cursor glow effect on cards ----
    document.querySelectorAll('.motivation-card, .theme-card, .objective-card, .dataset-card, .latent-card, .team-card, .algo-card').forEach(card => {
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            card.style.setProperty('--mouse-x', `${x}px`);
            card.style.setProperty('--mouse-y', `${y}px`);
            card.style.background = `radial-gradient(300px circle at ${x}px ${y}px, rgba(249, 115, 22, 0.06), rgba(255,255,255,0.03))`;
        });

        card.addEventListener('mouseleave', () => {
            card.style.background = 'rgba(255,255,255,0.03)';
        });
    });
});
