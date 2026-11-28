const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
const motionChoice = document.querySelector(".motion-choice");
const checkbox = motionChoice.querySelector("input");
const page = document.documentElement;
let scrollFrame;

/** Follow the system preference unless the visitor opts into motion in this tab. */
function updateMotion() {
  motionChoice.hidden = !reducedMotion.matches;
  const enabled = !reducedMotion.matches || checkbox.checked;
  page.classList.toggle("motion-enabled", enabled);
  if (!enabled) {
    for (const group of document.querySelectorAll(".section-entered")) group.classList.remove("section-entered");
  }
}

try {
  checkbox.checked = sessionStorage.getItem(checkbox.id) === "on";
} catch { /* The choice still works when browser storage is unavailable. */ }
checkbox.addEventListener("change", () => {
  updateMotion();
  try {
    sessionStorage.setItem(checkbox.id, checkbox.checked ? "on" : "off");
  } catch { /* Keep the choice for this page view. */ }
});
reducedMotion.addEventListener("change", updateMotion);
updateMotion();

/** Let the reader interrupt an arrow with the wheel, keyboard, touch or Back. */
function stopScrolling() {
  cancelAnimationFrame(scrollFrame);
}

/** Animate section arrows even in viewers that force native anchor scrolling to jump. */
function followArrow(event) {
  const link = event.currentTarget;
  const section = document.getElementById(link.hash.slice(1));
  if (!page.classList.contains("motion-enabled") || !section || event.button !== 0 ||
      event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;

  event.preventDefault();
  stopScrolling();
  const start = window.scrollY;
  const inset = parseFloat(getComputedStyle(page).scrollPaddingTop) || 0;
  const target = start + section.getBoundingClientRect().top - inset;
  const end = Math.max(0, Math.min(target, page.scrollHeight - page.clientHeight));
  if (location.hash !== link.hash) history.pushState(null, "", link.hash);
  section.focus({ preventScroll: true });
  const started = performance.now();

  /** Ease both ends of the scroll and finish immediately if motion is switched off. */
  function step(now) {
    const progress = page.classList.contains("motion-enabled") ? Math.min(1, Math.max(0, (now - started) / 580)) : 1;
    const eased = progress * progress * (3 - 2 * progress);
    window.scrollTo({ top: start + (end - start) * eased, behavior: "instant" });
    if (progress < 1) scrollFrame = requestAnimationFrame(step);
  }
  scrollFrame = requestAnimationFrame(step);
}

for (const link of document.querySelectorAll(".section-next")) link.addEventListener("click", followArrow);
for (const type of ["wheel", "touchstart", "pointerdown", "keydown", "popstate", "pagehide"]) {
  window.addEventListener(type, stopScrolling, { passive: true });
}

// Each group is visible by default, including when opened directly or printed.
if ("IntersectionObserver" in window) {
  const observer = new IntersectionObserver(entries => {
    for (const entry of entries) {
      if (entry.isIntersecting) {
        if (page.classList.contains("motion-enabled")) {
          entry.target.classList.add("section-entered");
        }
        observer.unobserve(entry.target);
      }
    }
  });
  const groups = document.querySelectorAll(
    ".spotlight > .content, .spotlight > figure, .river-details, .math-topic, .river-replay, " +
    ".spring-overview, .results, .verification, .technology"
  );
  for (const group of groups) observer.observe(group);
}
