import {
  Chart, BarController, BarElement, CategoryScale, LinearScale,
  LineController, LineElement, PointElement, Title, Tooltip
} from "chart.js";
import DataLabels from "chartjs-plugin-datalabels";

Chart.register(BarController, BarElement, CategoryScale, LinearScale,
  LineController, LineElement, PointElement, Title, Tooltip, DataLabels);

const styles = getComputedStyle(document.documentElement);
/** Read the same palette and typeface that style the rest of the page. */
function theme(name) {
  return styles.getPropertyValue(`--${name}`).trim();
}

const policyColors = [theme("baseline"), theme("controller")];
Chart.defaults.font.family = theme("sans");
Chart.defaults.font.size = 13;
Chart.defaults.color = theme("muted");
Chart.defaults.borderColor = theme("chart-grid");
Chart.defaults.animation = false;
Chart.defaults.maintainAspectRatio = false;
Chart.defaults.set("plugins.title", {
  display: true, align: "start", color: theme("ink"), font: { size: 14, weight: 600 }
});
Chart.defaults.set("plugins.tooltip", {
  backgroundColor: theme("deep"), titleColor: theme("paper"), bodyColor: theme("paper")
});
Chart.defaults.set("plugins.datalabels", {
  align: "right", anchor: "end", offset: 6, padding: 0, font: { weight: 600 }
});

/** Draw the two policies' changes in level during the same quarter-month. */
function lakeOntarioRise(names, rows) {
  return {
    type: "bar",
    data: {
      labels: names,
      datasets: [{ data: rows[0].values, backgroundColor: policyColors, maxBarThickness: 24 }]
    },
    options: {
      indexAxis: "y",
      layout: { padding: { right: 32 } },
      scales: {
        x: { beginAtZero: true, ticks: { maxTicksLimit: 5 } },
        y: { grid: { display: false }, ticks: { color: theme("ink") } }
      },
      plugins: {
        datalabels: { color: ctx => policyColors[ctx.dataIndex], formatter: value => value.toFixed(2) },
        tooltip: { callbacks: { label: ctx => `${ctx.raw.toFixed(2)} cm` } }
      }
    }
  };
}

/** Trace end-of-period storage and label each policy at its last point. */
function ottawaStorage(names, rows) {
  return {
    type: "line",
    data: {
      labels: rows.map(row => row.period),
      datasets: names.map((label, i) => ({
        label, data: rows.map(row => row.values[i]),
        borderColor: policyColors[i], backgroundColor: policyColors[i], borderWidth: 2.5,
        borderDash: i === 0 ? [6, 4] : [], pointStyle: i === 0 ? "rect" : "circle", pointRadius: 3.5
      }))
    },
    options: {
      layout: { padding: { right: 122 } },
      interaction: { mode: "index", intersect: false },
      scales: {
        x: {
          grid: { display: false },
          ticks: {
            autoSkip: false, maxRotation: 0,
            /** Stack the month over its quarter on narrow charts. */
            callback(index) {
              const period = rows[index].period;
              return this.chart.width < 400 ? period.split(" ") : period;
            }
          }
        },
        y: { ticks: { maxTicksLimit: 4 } }
      },
      plugins: {
        datalabels: {
          display: ctx => ctx.dataIndex === rows.length - 1,
          color: ctx => ctx.dataset.borderColor, formatter: (_, ctx) => ctx.dataset.label
        },
        tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: ${ctx.raw.toFixed(4)} billion m³` } }
      }
    }
  };
}

/** Use the full-precision values Hugo put in the figure's accessible data table. */
function chartConfig(figure) {
  const table = figure.querySelector("table");
  const names = Array.from(table.tHead.rows[0].cells).slice(1).map(th => th.textContent);
  const rows = Array.from(table.tBodies[0].rows, row => ({
    period: row.cells[0].textContent,
    values: Array.from(row.querySelectorAll("td"), cell => Number(cell.dataset.value))
  }));
  const config = figure.dataset.river === "lake-ontario" ? lakeOntarioRise(names, rows) : ottawaStorage(names, rows);
  config.options.plugins.title = { text: table.caption.textContent };
  return config;
}

/** Render a separate 2400 × 960 PNG so even a phone can export a poster figure. */
function posterPng(figure) {
  const canvas = document.createElement("canvas");
  canvas.width = 600;
  canvas.height = 240;
  const config = chartConfig(figure);
  config.options.responsive = false;
  config.options.devicePixelRatio = 4;
  const title = config.options.plugins.title;
  title.text = [figure.querySelector("h3").textContent, figure.querySelector(".figure-period").textContent, title.text];
  const chart = new Chart(canvas, config);
  chart.ctx.globalCompositeOperation = "destination-over";
  chart.ctx.fillStyle = theme("paper");
  chart.ctx.fillRect(0, 0, chart.width, chart.height);
  const image = chart.toBase64Image();
  chart.destroy();
  return image;
}

/** Keep the source table visible until the browser can draw its chart. */
function showFigure(figure) {
  const plot = figure.querySelector(".river-plot");
  const canvas = plot.querySelector("canvas");
  if (!canvas.getContext("2d")) return;
  const config = chartConfig(figure);
  plot.hidden = false;
  new Chart(canvas, config);
  figure.querySelector(".figure-data").open = false;
  const download = figure.querySelector(".figure-download");
  download.href = posterPng(figure);
  download.hidden = false;
}

document.fonts.ready.then(() => document.querySelectorAll(".river-graph").forEach(showFigure));
