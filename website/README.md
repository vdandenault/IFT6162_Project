# Lake Ontario–Ottawa website

One Hugo Story page for the 2017 and 2019 flood-year replays. Python exports the checked results, Hugo builds the page, and the browser draws the river figures with Chart.js.

## Preview

Use Node.js 24 LTS. After cloning the repository or pulling a changed `package-lock.json`, open macOS Terminal or Windows Command Prompt in the project repository and run

```sh
cd website
npm ci
```

In VS Code, open the project repository, select **04 - Website: preview Hugo Story** in Run and Debug, and press **F5**. Open the same address in every browser. Edits reload automatically. [Local website](http://127.0.0.1:1313/)

Profile 04 runs the package’s `dev` script on port 1313. If the preview is already running, reuse it. The red Stop button in VS Code ends the server. If another application owns port 1313, the Debug Console reports the conflict.


## Edit

| To change | Edit |
| --- | --- |
| Banner, spotlight introductions and next-section links | [data/story.yaml](data/story.yaml) |
| Opening finding and basin map | [banner.html](themes/hugo-story/layouts/_partials/banner.html) and [river-map.html](layouts/_partials/river-map.html) |
| Flood context, Plan 2014, simulator links and study constraints | [overview.html](layouts/_partials/overview.html) |
| River equations and the symbols printed below each one | [data/equations.yaml](data/equations.yaml) |
| Operator objectives and how each equation is used | [river-dynamics.html](layouts/_partials/river-dynamics.html) |
| May 2017 figure data and captions | [river-figure.html](layouts/_partials/river-figure.html) |
| River chart styles, tooltips and PNG export | [river-figures.js](assets/js/river-figures.js) |
| All page, map and chart colors | [assets/css/theme.css](assets/css/theme.css) |
| River chart placement and shared controls | [river-replay.html](layouts/_partials/river-replay.html) and [replay-controls.html](layouts/_partials/replay-controls.html) |
| Spring comparison, tradeoffs and verification | [spring-comparison.html](layouts/_partials/spring-comparison.html) and [replay.html](layouts/_partials/replay.html) |
| Full-year replay charts and interactions | [charts.js](assets/js/charts.js) and [replay.js](assets/js/replay.js) |
| Simulation and website technologies | [technology.html](layouts/_partials/technology.html) |

Hugo typesets the LaTeX at build time using its built-in KaTeX renderer. Edit an equation as you would in Overleaf, without the surrounding display delimiters. Each symbol definition is a pair of LaTeX and plain text. The two adapted Story templates are under [themes/hugo-story](themes/hugo-story/LICENSE.md). Photo credits appear in the flood photograph’s caption and in the footer.

Both rivers and Results share the selected year and MPC. The river plots follow one quarter-month cursor. The two May 2017 figures remain fixed worked examples.

## Check and build

```sh
node --run lint
node --run build
```

Lint runs ESLint and Stylelint. Hugo writes the static page and assets to `public/`. For a repository Pages site, pass its agreed base URL to the build, for example `node --run build -- --baseURL https://OWNER.github.io/REPOSITORY/`.

## Reproduce the two flood springs

The website reads precomputed closed-loop trajectories. Each Python run uses the current model state and available forecasts to choose a release, executes one quarter-month with realized inflows, then plans again. The year selector changes the matched disturbance scenario. The spring flood exposure figure in Results compares all three policies. The river plots keep filtered Plan 2014 beside the selected MPC.

Use a separate simulator checkout at [the revision used here](https://github.com/vdandenault/IFT6162_Project/tree/9cc82675ef5c95661b4973c38832d4fcbec010a0), with the simulator’s input data available. From that checkout’s root, in macOS Terminal or Git Bash, run the following with [uv](https://docs.astral.sh/uv/). Choose a fresh output directory outside the repository.

```sh
git -c core.autocrlf=false submodule update --init --recursive
uv sync --locked --group ontario
for arm in rule_curve external_mpc stackelberg_mpc; do
  uv run --locked --group ontario python -m reservoir_control.experiments.final_coupled_study \
    --mode development --event-only --recovery peak_protected \
    --controller "$arm" --output "../flood-year-runs/${arm}_events"
done
```

From the **website repository root**, run the exporter with Python 3.11 or newer. Set the first path to the run directory above and the repository argument to the simulator checkout.

```sh
python website/tools/export_replay.py ../flood-year-runs website/static/replay.json \
  --repository ../IFT6162_Project-vincent --revision 9cc82675ef5c95661b4973c38832d4fcbec010a0
```

The exporter checks source hashes, matched forcing and initial state, all 48 executed steps per year, and annual metrics. It prints the exact ordinary-versus-anticipating Lake Ontario release count for each year. At this revision the counts are **48/48 in 2017 and 48/48 in 2019**. The spring comparison uses all three policies’ April–June rows (quarter-months 13–24) for flood exposure, peaks and economic impacts. The Ottawa storage comparison uses the last step of the year. Only the derived JSON goes into the website.

The two river figures update on the next Hugo build. Hugo reads the 2017 replay into a small HTML table for each chart. Chart.js plots the same cells in the browser. **View chart data** opens the values, and **Download figure (PNG)** renders a 2400 × 960 image with the year, units and policy labels. Check the May 2017 captions when replacing the runs.
