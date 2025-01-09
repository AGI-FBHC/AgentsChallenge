document.addEventListener('DOMContentLoaded', function() {
    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });

    // 获取当前页面的路径
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.nav-links a');

    // 设置当前页面的导航链接高亮
    navLinks.forEach(link => {
        // 移除所有现有的 aria-current 属性
        link.removeAttribute('aria-current');
        
        // 获取链接的 href 属性
        const href = link.getAttribute('href');
        
        // 如果链接的 href 匹配当前路径，设置为当前页面
        if (href === currentPath || 
            (currentPath === '/' && href === '/index.html') ||
            (currentPath.endsWith(href))) {
            link.setAttribute('aria-current', 'page');
        }
    });

    // 如果页面有锚点导航，保留滚动监听功能
    const sections = document.querySelectorAll('.section');
    if (sections.length > 0) {
        window.addEventListener('scroll', () => {
            let current = '';
            const scrollPosition = window.scrollY + window.innerHeight / 3;

            sections.forEach(section => {
                const sectionTop = section.offsetTop;
                const sectionHeight = section.offsetHeight;
                
                if (scrollPosition >= sectionTop && scrollPosition < sectionTop + sectionHeight) {
                    current = section.getAttribute('id');
                }
            });

            if (current) {
                navLinks.forEach(link => {
                    link.classList.remove('active');
                    if (link.getAttribute('href') === '#' + current) {
                        link.classList.add('active');
                    }
                });
            }
        });
    }
}); 