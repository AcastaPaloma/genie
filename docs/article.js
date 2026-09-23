const contents = document.querySelector('.contents');
const compactLayout = window.matchMedia('(max-width: 800px)');
const links = [...document.querySelectorAll('.contents nav a')];
const headings = links.map((link) => document.getElementById(link.hash.slice(1)));

// Native anchors keep deep links, browser history, and no-JavaScript reading intact.
function setContentsLayout() {
  contents.open = !compactLayout.matches;
}
setContentsLayout();
compactLayout.addEventListener('change', setContentsLayout);

for (const heading of headings) heading.tabIndex = -1;
for (const link of links) {
  link.addEventListener('click', () => {
    if (compactLayout.matches) contents.open = false;
    document.getElementById(link.hash.slice(1)).focus({ preventScroll: true });
  });
}

let scheduled = false;
let activeLink = links[0];
function updateCurrentSection() {
  scheduled = false;
  const threshold = compactLayout.matches ? 125 : 100;
  let current = 0;
  for (let index = 0; index < headings.length; index += 1) {
    if (headings[index].getBoundingClientRect().top <= threshold) current = index;
    else break;
  }
  // The final section can be shorter than the viewport and never reach the threshold.
  if (window.scrollY + window.innerHeight >= document.documentElement.scrollHeight - 2) {
    current = headings.length - 1;
  }
  const next = links[current];
  if (next !== activeLink) {
    activeLink.removeAttribute('aria-current');
    next.setAttribute('aria-current', 'location');
    activeLink = next;
  }
}
function scheduleUpdate() {
  if (!scheduled) {
    scheduled = true;
    requestAnimationFrame(updateCurrentSection);
  }
}
window.addEventListener('scroll', scheduleUpdate, { passive: true });
window.addEventListener('resize', scheduleUpdate, { passive: true });
window.addEventListener('pageshow', scheduleUpdate);
document.fonts.ready.then(scheduleUpdate);
