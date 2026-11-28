const svgNamespace = "http://www.w3.org/2000/svg";
const levelFields = new Set(["lake_ontario_level_m", "lake_st_louis_level_m"]);

/** Add an SVG element using the coordinates and accessible text for this figure. */
function draw(parent, tag, attributes, text = "") {
  const node = document.createElementNS(svgNamespace, tag);
  for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, value);
  if (text) node.textContent = text;
  parent.append(node);
  return node;
}

/** Keep a site's vertical scale fixed across both years and all three policies. */
export function scaleFor(data, field) {
  const values = [];
  for (const year of Object.values(data.years)) {
    for (const policy of Object.values(year.policies)) {
      values.push(...policy[field]);
      if (field === "lake_st_louis_level_m") values.push(...policy.lake_st_louis_limit_m);
    }
  }
  if (field === "lake_ontario_level_m") values.push(data.lake_ontario_limit_m);
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const span = hi - lo || 1;
  const power = 10 ** Math.floor(Math.log10(span / 4));
  const step = [1, 2, 2.5, 5, 10].find(value => value * power >= span / 4) * power;
  return {
    low: Math.floor((lo - span * .02) / step) * step,
    high: Math.ceil((hi + span * .02) / step) * step,
    step
  };
}

/** Connect measured quarter-month levels, or hold releases and limits over each step. */
function linePath(values, x, y, stepped = false) {
  let path = `M${x(0)} ${y(values[0])}`;
  for (let i = 1; i < values.length; i += 1) {
    path += stepped ? ` H${x(i)} V${y(values[i])}` : ` L${x(i)} ${y(values[i])}`;
  }
  return path;
}

/** Attach names to the final points and separate the labels when curves coincide. */
function labelLines(svg, rows, right, top, bottom) {
  const [upper, lower] = [...rows].sort((a, b) => a.y - b.y);
  const upperLabel = Math.max(top + 6, Math.min(upper.y, lower.y - 16, bottom - 16));
  const lowerLabel = Math.min(bottom, Math.max(lower.y, upperLabel + 16));
  for (const [row, labelY] of [[upper, upperLabel], [lower, lowerLabel]]) {
    draw(svg, "path", { d: `M${right} ${row.y} l5 0 L${right + 10} ${labelY}`, class: `leader ${row.style}` });
    draw(svg, "text", { x: right + 13, y: labelY + 4, class: `trace-label ${row.style}-label` }, row.label);
  }
}

/** Draw one chart and the exact values at the selected quarter-month. */
export function renderChart(figure, scale, data, baseline, policy, label, quarter) {
  const { field, unit } = figure.dataset;
  const svg = figure.querySelector("svg");
  const isLevel = levelFields.has(field);
  const width = Math.max(240, Math.round(svg.getBoundingClientRect().width));
  const height = Math.max(150, Math.round(svg.getBoundingClientRect().height));
  const left = isLevel ? 39 : 48;
  const right = width - (width < 410 ? 104 : 138);
  const top = 15;
  const bottom = height - 26;
  const count = baseline[field].length;
  const x = index => left + index * (right - left) / (count - 1);
  const y = value => bottom - (value - scale.low) / (scale.high - scale.low) * (bottom - top);
  const format = value => value.toLocaleString("en-CA", {
    minimumFractionDigits: isLevel ? 2 : 0,
    maximumFractionDigits: isLevel ? 2 : 0
  });
  svg.replaceChildren();
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  draw(svg, "rect", { x: x(11.5), y: top, width: x(23.5) - x(11.5), height: bottom - top, class: "spring-window" });
  for (let tick = scale.low; tick <= scale.high + scale.step / 2; tick += scale.step) {
    draw(svg, "line", { x1: left, y1: y(tick), x2: right, y2: y(tick), class: "grid" });
    draw(svg, "text", { x: left - 5, y: y(tick) + 4, "text-anchor": "end" },
      Number(tick.toFixed(2)).toLocaleString("en-CA"));
  }
  for (const [index, month] of [[0, "Jan"], [12, "Apr"], [24, "Jul"], [36, "Oct"]]) {
    draw(svg, "text", { x: x(index), y: bottom + 20, "text-anchor": "middle" }, month);
  }
  let limits = "";
  if (field === "lake_ontario_level_m") {
    const limit = data.lake_ontario_limit_m;
    draw(svg, "path", { d: `M${left} ${y(limit)} H${right}`, class: "threshold" });
    limits = ` · Flood threshold ${format(limit)} m`;
  }
  if (field === "lake_st_louis_level_m") {
    const ref = baseline.lake_st_louis_limit_m;
    const target = policy.lake_st_louis_limit_m;
    const different = ref.some((v, i) => v !== target[i]);
    if (different) draw(svg, "path", { d: linePath(ref, x, y, true), class: "threshold baseline-limit" });
    draw(svg, "path", { d: linePath(target, x, y, true), class: "threshold" });
    limits = ref[quarter] === target[quarter] ? ` · Stage target ${format(target[quarter])} m` :
      ` · Stage targets ${format(ref[quarter])} m (Plan 2014) and ${format(target[quarter])} m (${label})`;
  }
  const rows = [
    { values: policy[field], label: width < 410 ? label.replace(" MPC", "") : label, style: "controller" },
    { values: baseline[field], label: "Plan 2014", style: "baseline" }
  ];
  for (const row of rows) {
    draw(svg, "path", { d: linePath(row.values, x, y, !isLevel), class: `trace ${row.style}` });
    row.y = y(row.values[count - 1]);
  }
  labelLines(svg, rows, right, top, bottom);
  draw(svg, "line", { x1: x(quarter), y1: top, x2: x(quarter), y2: bottom, class: "cursor" });
  for (const row of [...rows].reverse()) {
    draw(svg, "circle", { cx: x(quarter), cy: y(row.values[quarter]), r: row.style === "baseline" ? 4 : 3, class: `${row.style}-dot` });
  }
  figure.querySelector(".chart-values").textContent =
    `Plan 2014 ${format(baseline[field][quarter])} · ${label} ${format(policy[field][quarter])} ${unit}${limits}`;
}
