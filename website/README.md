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

## Check and build

```sh
node --run lint
node --run build
```

Lint runs ESLint and Stylelint. Hugo writes the static page and assets to `public/`. For a repository Pages site, pass its agreed base URL to the build, for example `node --run build -- --baseURL https://OWNER.github.io/REPOSITORY/`.
