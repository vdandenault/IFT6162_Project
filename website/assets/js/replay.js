import { renderChart, scaleFor } from "./charts.js";

const replay = document.querySelector("#replay");
const controls = [...document.querySelectorAll("[data-replay-controls]")];
const timelines = [...document.querySelectorAll(".timeline")];
const charts = [...document.querySelectorAll(".chart")];
const names = { ordinary: "Ordinary MPC", anticipating: "Anticipating MPC" };
const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const state = { year: "2017", policy: "ordinary", quarter: 13 };
let timer;

/** Give both rivers the same play or pause action. */
function showPlaying(playing) {
  for (const button of document.querySelectorAll("[data-play]")) {
    button.textContent = playing ? "Pause" : "Play";
    button.setAttribute("aria-pressed", String(playing));
  }
}

/** Stop the clock before changing cases or inspecting a quarter-month manually. */
function pause() {
  clearInterval(timer);
  timer = undefined;
  showPlaying(false);
}

/** Keep the year and method paired with the same saved baseline throughout the page. */
function selection(data) {
  const { year, policy } = state;
  const { policies, design } = data.years[year];
  return { year, policies, design, baseline: policies.plan2014, chosen: policies[policy], name: names[policy] };
}

/** Format displayed estimates at a stated precision without changing the stored values. */
function number(value, digits) {
  return value.toLocaleString("en-CA", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

/** Fill a table row, retaining the original units until the display conversion. */
function metricRow(body, title, values, digits, divisor = 1) {
  const row = body.insertRow();
  const heading = document.createElement("th");
  heading.scope = "row";
  heading.textContent = title;
  row.append(heading);
  for (const value of values) row.insertCell().textContent = number(value / divisor, digits);
}

/** Label a group of outcomes with its measurement window. */
function metricGroup(body, title) {
  body.replaceChildren();
  const heading = document.createElement("th");
  heading.colSpan = 3;
  heading.scope = "rowgroup";
  heading.textContent = title;
  body.insertRow().append(heading);
}

/** Compare both complete trajectories and describe the change in selected spring exposure. */
function updateResults({ year, design, policies, baseline, chosen, name }) {
  const changes = policies.ordinary.lake_ontario_release_m3_s.filter(
    (release, i) => release !== policies.anticipating.lake_ontario_release_m3_s[i]
  ).length;
  const overlap = changes === 0
    ? `Across the schedules tested, both MPCs chose the same Lake Ontario release in all ${policies.ordinary.lake_ontario_release_m3_s.length} quarter-months of ${year}.`
    : `The two MPCs chose different Lake Ontario releases in ${changes} quarter-months of ${year}.`;
  const pattern = { seasonal_wet: "seasonally wet", steady: "steady" }[design] ?? design;
  document.querySelector("#scenario-note").textContent =
    `${year} Lake Ontario conditions with a ${pattern} synthetic Ottawa inflow.`;
  const difference = chosen.spring.lake_ontario_exposure_m_qm - baseline.spring.lake_ontario_exposure_m_qm;
  const change = difference === 0 ? "matched Plan 2014 for spring Lake Ontario flood exposure" :
    `${difference < 0 ? "reduced" : "increased"} spring Lake Ontario flood exposure by ${number(Math.abs(difference), 3)} m·quarter-months relative to Plan 2014`;
  const withinTarget = [baseline, chosen].every(p => p.spring.lake_st_louis_periods_above === 0);
  const downstream = withinTarget ? " Both stayed within the Pointe-Claire target throughout the spring." : "";
  document.querySelector("#finding").textContent = `${overlap} ${name} ${change}.${downstream}`;
  document.querySelector("#selected-name").textContent = name;
  document.querySelector("#metrics-caption").textContent = `Modeled outcomes in ${year}`;

  const spring = document.querySelector("#spring-metrics");
  metricGroup(spring, "April–June");
  for (const [key, title, digits, divisor] of [
    ["lake_ontario_peak_m", "Peak Lake Ontario level (m)", 3],
    ["lake_st_louis_peak_m", "Peak Pointe-Claire level (m)", 2],
    ["lake_ontario_exposure_m_qm", "Lake Ontario flood exposure (m·quarter-months)", 3],
    ["lake_st_louis_exposure_m_qm", "Pointe-Claire flood exposure (m·quarter-months)", 3],
    ["lake_ontario_periods_above", "Lake Ontario periods above the study level", 0],
    ["lake_st_louis_periods_above", "Pointe-Claire periods above the study level", 0],
    ["upstream_peak_buildings", "Peak affected buildings, upstream (estimated)", 0],
    ["downstream_peak_buildings", "Peak affected buildings, downstream (estimated)", 0],
    ["hydropower_value_usd", "Hydropower value (million USD)", 3, 1e6],
    ["boating_cost_usd", "Recreational boating cost (million USD)", 3, 1e6]
  ]) metricRow(spring, title, [baseline.spring[key], chosen.spring[key]], digits, divisor);

  const annual = document.querySelector("#annual-metrics");
  metricGroup(annual, "Full model year");
  for (const [key, title, digits, divisor] of [
    ["annual_cost", "Operating cost (objective units)", 3],
    ["annual_ottawa_end_storage_m3", "Ottawa year-end storage (billion m³)", 3, 1e9],
    ["annual_ottawa_service_shortfall_million_m3", "Ottawa service shortfall (million m³)", 1]
  ]) metricRow(annual, title, [baseline[key], chosen[key]], digits, divisor);
}

/** Expose all three panels together, with a local message if loading fails. */
function showPanels(ready) {
  for (const panel of document.querySelectorAll("[data-replay-panel]")) panel.hidden = !ready;
  for (const status of document.querySelectorAll(".replay-status")) {
    status.hidden = ready;
    if (!ready) status.textContent = "The data could not be loaded. Reload this page to try again.";
  }
}

/** Bind the river and result controls to one matched pair of verified trajectories. */
async function startReplay() {
  const response = await fetch(replay.dataset.source);
  if (!response.ok) throw new Error(`Replay data returned ${response.status}`);
  const data = await response.json();
  const scales = charts.map(figure => scaleFor(data, figure.dataset.field));

  /** Move both cursors and read the coordinates from the same quarter-month. */
  function updateQuarter() {
    const { year, baseline, chosen, name } = selection(data);
    const index = state.quarter - 1;
    const date = `${months[Math.floor(index / 4)]} ${year} · ${index % 4 + 1}/4`;
    for (const timeline of timelines) {
      const slider = timeline.querySelector("input");
      slider.value = state.quarter;
      slider.setAttribute("aria-valuetext", `${date}, quarter-month ${state.quarter} of 48`);
      timeline.querySelector("output").value = date;
    }
    const shortName = name.split(" ")[0];
    charts.forEach((figure, i) => renderChart(figure, scales[i], data, baseline, chosen, shortName, index));
  }

  /** Carry a changed case to every section without losing the selected quarter-month. */
  function updateCase() {
    pause();
    for (const form of controls) {
      form.elements.year.value = state.year;
      form.elements.policy.value = state.policy;
    }
    for (const spring of document.querySelectorAll("[data-flood-year]")) {
      spring.hidden = spring.dataset.floodYear !== state.year;
    }
    updateResults(selection(data));
    updateQuarter();
  }

  /** Play the full year from the current step, restarting after the last step. */
  function togglePlay() {
    if (timer !== undefined) { pause(); return; }
    if (state.quarter === 48) state.quarter = 1;
    showPlaying(true);
    updateQuarter();
    timer = setInterval(() => {
      state.quarter += 1;
      updateQuarter();
      if (state.quarter === 48) pause();
    }, 700);
  }

  for (const form of controls) {
    for (const radio of form.querySelectorAll("input[type=radio]")) {
      radio.addEventListener("change", () => {
        state.year = form.elements.year.value;
        state.policy = form.elements.policy.value;
        updateCase();
      });
    }
  }
  for (const timeline of timelines) {
    timeline.querySelector("input").addEventListener("input", event => {
      pause();
      state.quarter = Number(event.target.value);
      updateQuarter();
    });
    timeline.querySelector("button").addEventListener("click", togglePlay);
  }
  document.addEventListener("visibilitychange", () => { if (document.hidden) pause(); });
  window.addEventListener("pagehide", pause);
  showPanels(true);
  updateCase();
  const observer = new ResizeObserver(updateQuarter);
  for (const figure of charts) observer.observe(figure.querySelector("svg"));
}

startReplay().catch(error => {
  pause();
  showPanels(false);
  console.error(error);
});
