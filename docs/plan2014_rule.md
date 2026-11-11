

Let :

\[
    \begin{aligned}
        t &: && \text{Simulation timestep in quarter-months}; \\
        Q_{t} &: && \text{Volumetric discharge } (m^3/s); \\
        Q_{t}^{out} &: && \text{Outflow } (m^3/s); \\
        Q_{t}^{in} &: && \text{Inflow } (m^3/s); \\
        R_{t} &: && \text{Hydraulic roughness coefficient } ; \\
        \eta_{t} &: && \text{Tidal water-level variation } (m); \\



    \end{aligned}
\]

<br>
<details closed>
<summary><code>Variables</code></summary>
<br> 

| . | Dataset column | Domain | Description | 
| --- | --- | --- | --- |
| $t$ | Sim | $\{ 0, \dots, T-1 \}$ |
| $k$ | QM | |
| $m$ | Year | |
| $y$ | Month | |
| $h_{t}$ | ontLevelBOQ | \{ 74.55 \} | Lake Ontario (LO) stage beginning of quarter |
| $h_{t}^{obs}$ | ontLevel | | LO observed stage. |
| $Q_{t}^{obs}$ | ontFlow | | LO observed discharge. |
| $Q_{t}^{in}$ | ontNBS | | LO inflow (net total supply) |
| $Q_{t}^{out}$ | ontNTS | | Regulated LO outflow |
| $Q_{t}^{Erie}$ | erieOut | | LO inflow from Lake Erie and connecting channels |
| $Q_{t}^{SLON}$ | stlouisontOut | | LO outlow to Lac Saint-Louis |
| $Q_{t}^{DP}$ | desprairiesOut | | LO outlow to Des Prairies River |
| $Q_{t}^{SF}$ | stfrancoisOut | | LO outlow to Saint-François River |
| $Q_{t}^{SM}$ | stmauriceOut | | LO outlow to Saint-Maurice River |
| $Q_{t}^{Ric}$ | richelieuOut | | LO outlow to Richelieu River |
| $I_{t}^{ice}$ | iceInd | $\{ 0, 1, 2\}$ | Categorical ice condition |
| $\eta_{t}$ | tidalInd |
| $R_{t}^{LS}$ | longsaultR | | Roughness coefficient at Long Sault Dam |
| $R_{t}^{SH}$ | saundershwR | | Roughness coefficient at the headwaters of Moses-Saunders Dam |
| $R_{t}^{PC}$ | ptclaireR | | Roughness coefficient at Pointe-Claire |
| $R_{t}^{Og}$ | ogdensburgR | | Roughness coefficient at Ogdensburg |
| $R_{t}^{C}$ | cardinalR | | Roughness coefficient at Cardinal |
| $R_{t}^{IH}$ | iroquoishwR | | Roughness coefficient at the headwaters of Iroquois Dam |
| $R_{t}^{IT}$ | iroquoistwR | | Roughness coefficient at the tailwaters of Iroquois Dam |
| $R_{t}^{Mo}$ | morrisburgR | | Roughness coefficient at Morrisburg |
| $R_{t}^{ST}$ | saunderstwR | | Roughness coefficient at the tailwaters of Moses-Saunders Dam |
| $R_{t}^{CW}$ | cornwallR | | Roughness coefficient at Cornwall |
| $R_{t}^{Su}$ | summerstownR | | Roughness coefficient at Summerstown |
| $R_{t}^{J}$ | jetty1R | | Roughness coefficient at Jetty 1 |
| $R_{t}^{Va}$ | varennesR | | Roughness coefficient at Varennes |
| $R_{t}^{So}$ | sorelR | | Roughness coefficient at Sorel |
| $R_{t}^{SP}$ | stpierreR | | Roughness coefficient at Saint-Pierre |
| $R_{t}^{TR}$ | threeriversR | | Roughness coefficient at Trois-Rivières |
| $R_{t}^{B}$ | batiscanR | | Roughness coefficient at Batiscan |
</details>

<br> 
Time Indexing
<div style="margin-left: 24px;">
With year index $y$, month index $m$ within the year, quarter-month index $q$ within the month, quarter-month index $k$ within the year, and simulation timestep $t$.

$$
t = 48y + 4(m-1) + (q-1) = 48y + k - 1
$$
</div>

<br>
LO Net Total Supply (NTS)
\[ Q_t^{in} := Q_t^{NTS} \\ Q_t^{in} = Q_t^{NBS} + Q_t^{Erie}\]

<br>
Plan2014 Decision Rule




