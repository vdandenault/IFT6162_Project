

Let :

\[
    \begin{aligned}
        t &: && \text{Simulation timestep in quarter-months}; \\
        h_t&: && \text{reservoir stage } (m) \textit{ Syn. } \text{level};  \\
        h_w &: && \text{Crest elevation } (m);\\
        Q_{t} &: && \text{Volumetric discharge } (m^3/s); \\
        Q_{t}^{out} &: && \text{Outflow} (m^3/s) \textit{ Syn. } \text{release, discharge}; \\
        Q_{t}^{in} &: && \text{Inflow } (m^3/s); \\
        R_{t} &: && \text{Hydraulic roughness coefficient } ; \\
        \eta_{t} &: && \text{Tidal water-level variation } (m); \\



    \end{aligned}
\]

<br>
<details closed>
<summary><code>Variables</code></summary>
<br> 

| Notation | Dataset column | Domain | Description |
|---|---|---|---|
| $t$ | Sim | $\{0,\ldots,T-1\}$ | Simulation timestep |
| $y$ | Year | $\mathbb{N}$ | Calendar year |
| $m$ | Month | $\{1,\ldots,12\}$ | Month within the year |
| $k$ | QM | $\{1,\ldots,48\}$ | Quarter-month within the year |
| $h_t^{\mathrm{LO}}$ | ontLevelBOQ | $\mathbb{R}$ | Lake Ontario stage at the beginning of the quarter-month |
| $h_t^{\mathrm{LO,EOQ}}$ | ontLevelEOQ | $\mathbb{R}$ | Lake Ontario stage at the end of the quarter-month |
| $h_t^{\mathrm{LO,MOQ}}$ | ontLevelMOQ | $\mathbb{R}$ | Mean Lake Ontario stage during the quarter-month |
| $Q_t^{\mathrm{NBS,obs}}$ | ontNBS | $\mathbb{R}$ | Observed Lake Ontario net basin supply |
| $Q_t^{\mathrm{NTS,obs}}$ | ontNTS | $\mathbb{R}$ | Observed Lake Ontario net total supply |
| $Q_t^{\mathrm{out}}$ | ontFlow | $\mathbb{R}_{\geq 0}$ | Regulated Lake Ontario discharge |
| $Q_t^{\mathrm{Erie,obs}}$ | erieOut | $\mathbb{R}_{\geq 0}$ | Observed inflow from Lake Erie through connecting channels |
| $Q_t^{\mathrm{SLON,obs}}$ | stlouisontOut | $\mathbb{R}_{\geq 0}$ | Observed Ottawa River contribution to Lake Saint-Louis |
| $Q_t^{\mathrm{DP,obs}}$ | desprairiesOut | $\mathbb{R}_{\geq 0}$ | Observed Rivière des Prairies discharge |
| $Q_t^{\mathrm{SF,obs}}$ | stfrancoisOut | $\mathbb{R}_{\geq 0}$ | Observed Saint-François River discharge |
| $Q_t^{\mathrm{SM,obs}}$ | stmauriceOut | $\mathbb{R}_{\geq 0}$ | Observed Saint-Maurice River discharge |
| $Q_t^{\mathrm{Ric,obs}}$ | richelieuOut | $\mathbb{R}_{\geq 0}$ | Observed Richelieu River discharge |
| $Q_t^{\mathrm{SL}}$ | stlouisFlow | $\mathbb{R}_{\geq 0}$ | Simulated Lake Saint-Louis discharge |
| $h_t^{\mathrm{Kin}}$ | kingstonLevel | $\mathbb{R}$ | Kingston stage |
| $h_t^{\mathrm{AB}}$ | alexbayLevel | $\mathbb{R}$ | Alexandria Bay stage |
| $h_t^{\mathrm{BV}}$ | brockvilleLevel | $\mathbb{R}$ | Brockville stage |
| $h_t^{\mathrm{Og}}$ | ogdensburgLevel | $\mathbb{R}$ | Ogdensburg stage |
| $h_t^{\mathrm{Car}}$ | cardinalLevel | $\mathbb{R}$ | Cardinal stage |
| $h_t^{\mathrm{Ir,HW}}$ | iroquoishwLevel | $\mathbb{R}$ | Iroquois Dam headwater stage |
| $h_t^{\mathrm{Ir,TW}}$ | iroquoistwLevel | $\mathbb{R}$ | Iroquois Dam tailwater stage |
| $h_t^{\mathrm{Mor}}$ | morrisburgLevel | $\mathbb{R}$ | Morrisburg stage |
| $h_t^{\mathrm{LS}}$ | longsaultLevel | $\mathbb{R}$ | Long Sault stage |
| $h_t^{\mathrm{Sau,HW}}$ | saundershwLevel | $\mathbb{R}$ | Moses-Saunders Dam headwater stage |
| $h_t^{\mathrm{Sau,TW}}$ | saunderstwLevel | $\mathbb{R}$ | Moses-Saunders Dam tailwater stage |
| $h_t^{\mathrm{Cor}}$ | cornwallLevel | $\mathbb{R}$ | Cornwall stage |
| $h_t^{\mathrm{Sum}}$ | summerstownLevel | $\mathbb{R}$ | Summerstown stage |
| $h_t^{\mathrm{LB}}$ | lerybeauharnoisLevel | $\mathbb{R}$ | Léry–Beauharnois stage |
| $h_t^{\mathrm{PC}}$ | ptclaireLevel | $\mathbb{R}$ | Pointe-Claire stage |
| $h_t^{\mathrm{J1}}$ | jetty1Level | $\mathbb{R}$ | Montreal Harbour Jetty 1 stage |
| $h_t^{\mathrm{SLam}}$ | stlambertLevel | $\mathbb{R}$ | Saint-Lambert stage |
| $h_t^{\mathrm{Var}}$ | varennesLevel | $\mathbb{R}$ | Varennes stage |
| $h_t^{\mathrm{Sor}}$ | sorelLevel | $\mathbb{R}$ | Sorel stage |
| $h_t^{\mathrm{LSP}}$ | lacstpierreLevel | $\mathbb{R}$ | Lac Saint-Pierre stage |
| $h_t^{\mathrm{Mas}}$ | maskinongeLevel | $\mathbb{R}$ | Maskinongé stage |
| $h_t^{\mathrm{TR}}$ | troisrivieresLevel | $\mathbb{R}$ | Trois-Rivières stage |
| $h_t^{\mathrm{Bat}}$ | batiscanLevel | $\mathbb{R}$ | Batiscan stage |
| $I_t^{\mathrm{ice}}$ | iceInd | $\{0,1,2\}$ | categorical ice condition |
| $\eta_t^{\mathrm{obs}}$ | tidalInd | $\mathbb{R}$ | Observed tidal forcing index |
| $R_t^{\mathrm{LS}}$ | longsaultR | $\mathbb{R}_{>0}$ | Hydraulic roughness factor at Long Sault |
| $R_t^{\mathrm{Sau,HW}}$ | saundershwR | $\mathbb{R}_{>0}$ | Hydraulic roughness factor at Moses-Saunders Dam headwater |
| $R_t^{\mathrm{PC}}$ | ptclaireR | $\mathbb{R}_{>0}$ | Hydraulic roughness factor at Pointe-Claire |
| $R_t^{\mathrm{Og}}$ | ogdensburgR | $\mathbb{R}_{>0}$ | Hydraulic roughness factor at Ogdensburg |
| $R_t^{\mathrm{Car}}$ | cardinalR | $\mathbb{R}_{>0}$ | Hydraulic roughness factor at Cardinal |
| $R_t^{\mathrm{Ir,HW}}$ | iroquoishwR | $\mathbb{R}_{>0}$ | Hydraulic roughness factor at Iroquois Dam headwater |
| $R_t^{\mathrm{Ir,TW}}$ | iroquoistwR | $\mathbb{R}_{>0}$ | Hydraulic roughness factor at Iroquois Dam tailwater |
| $R_t^{\mathrm{Mor}}$ | morrisburgR | $\mathbb{R}_{>0}$ | Hydraulic roughness factor at Morrisburg |
| $R_t^{\mathrm{Sau,TW}}$ | saunderstwR | $\mathbb{R}_{>0}$ | Hydraulic roughness factor at Moses-Saunders Dam tailwater |
| $R_t^{\mathrm{Cor}}$ | cornwallR | $\mathbb{R}_{>0}$ | Hydraulic roughness factor at Cornwall |
| $R_t^{\mathrm{Sum}}$ | summerstownR | $\mathbb{R}_{>0}$ | Hydraulic roughness factor at Summerstown |
| $R_t^{\mathrm{J1}}$ | jetty1R | $\mathbb{R}_{>0}$ | Hydraulic roughness factor at Montreal Harbour Jetty 1 |
| $R_t^{\mathrm{Var}}$ | varennesR | $\mathbb{R}_{>0}$ | Hydraulic roughness factor at Varennes |
| $R_t^{\mathrm{Sor}}$ | sorelR | $\mathbb{R}_{>0}$ | Hydraulic roughness factor at Sorel |
| $R_t^{\mathrm{LSP}}$ | stpierreR | $\mathbb{R}_{>0}$ | Hydraulic roughness factor at Lac Saint-Pierre |
| $R_t^{\mathrm{TR}}$ | threeriversR | $\mathbb{R}_{>0}$ | Hydraulic roughness factor at Trois-Rivières |
| $R_t^{\mathrm{Bat}}$ | batiscanR | $\mathbb{R}_{>0}$ | Hydraulic roughness factor at Batiscan |
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

<br>
Problem
\[
    x_t = (
        I_t^{ice}, 
        Q_t^{\mathrm{SL}}, 
        h_t^{\mathrm{Kin}},
        h_t^{\mathrm{AB}},
        h_t^{\mathrm{BV}},
        h_t^{\mathrm{Og}},
        h_t^{\mathrm{Car}},
        h_t^{\mathrm{Ir,HW}},
        h_t^{\mathrm{Ir,TW}},
        h_t^{\mathrm{Mor}},
        h_t^{\mathrm{LS}},
        h_t^{\mathrm{Sau,HW}},
        h_t^{\mathrm{Sau,TW}},
        h_t^{\mathrm{Cor}},
        h_t^{\mathrm{Sum}},
        h_t^{\mathrm{LB}},
        h_t^{\mathrm{PC}},
        h_t^{\mathrm{J1}},
        h_t^{\mathrm{SLam}},
        h_t^{\mathrm{Var}},
        h_t^{\mathrm{Sor}},
        h_t^{\mathrm{LSP}},
        h_t^{\mathrm{Mas}},
        h_t^{\mathrm{TR}},
        h_t^{\mathrm{Bat}}
    )\\
\]

Bolza Problem
<!-- Verify that Q_t^{out} is really the control here and its not hiding env transition. -->


## Controlled Lake Ontario–St. Lawrence dynamics

We discretize time into 48 quarter-months per year. With zero-based year index $y$, month $m\in\{1,\ldots,12\}$ and quarter-month $q\in\{1,\ldots,4\}$, the annual quarter-month index and simulation index are

$$
k=4(m-1)+q,\qquad t=48y+k-1,\qquad q=1+((k-1)\bmod 4).
$$

Let $h_t$ be the beginning-of-quarter Lake Ontario stage, $N_t$ the quarter-monthly net total supply ($\mathrm{m}^3\,\mathrm{s}^{-1}$), and $u_t$ an additive discharge adjustment in units of $10\,\mathrm{m}^3\,\mathrm{s}^{-1}$. The nominal hydraulic rating relation and water balance are

$$
\begin{aligned}
\bar h_t &= \tfrac12(h_t+h_{t+1}),\\
r_t &= \frac{A_w}{10}(\bar h_t-H_0)^{3/2}+u_t,\\
h_{t+1} &= h_t+\frac{N_t-10r_t}{c}.
\end{aligned}
$$

Here $r_t$ is total release in units of $10\,\mathrm{m}^3\,\mathrm{s}^{-1}$, and

$$
\begin{aligned}
A_w &= 555.823,\qquad H_0=69.509\ \mathrm{m},\\
\Delta t &= 657450\ \mathrm{s},\qquad A_{\mathrm{ON}}=1.9526265\times10^{10}\ \mathrm{m}^2,\\
c &= \frac{A_{\mathrm{ON}}}{\Delta t}=29700\ \mathrm{m}^2\mathrm{s}^{-1}.
\end{aligned}
$$

Here $A_w$ is expressed in SI-compatible units ($\mathrm{m}^{3/2}\,\mathrm{s}^{-1}$) and is compatible with a weir-like coefficient $A_w\simeq\frac23 CL\sqrt{2g}$. The hydraulic rating relation is the adopted pre-project rating relation, not an assertion that release is uncontrolled. All hydraulic evaluations assume $\bar h_t\ge H_0$.

Combining the rating relation and water balance gives the **implicit midpoint transition**

$$
\boxed{
h_{t+1}=h_t+\frac{1}{c}\left[
N_t-A_w\left(\frac{h_t+h_{t+1}}{2}-H_0\right)^{3/2}-10u_t
\right].
}
$$

Setting $\rho_t=\sqrt{(h_t+h_{t+1})/2-H_0}$ yields

$$
\rho_t^3+a\rho_t^2-b_t=0,\qquad
a=\frac{2c}{A_w},\qquad
b_t=\frac{N_t-10u_t+2c(h_t-H_0)}{A_w}.
$$

For $b_t\ge0$, the nonnegative hydraulic-head root is

$$
\begin{aligned}
\eta_t &= \frac{27b_t}{2a^3}-1,\\
\rho_t &= \frac a3\bigl(2\chi(\eta_t)-1\bigr),\\
\chi(\eta)&=
\begin{cases}
\cos\!\left(\tfrac13\arccos\eta\right), &-1\le\eta\le1,\\
\cosh\!\left(\tfrac13\operatorname{arcosh}\eta\right), &\eta>1,
\end{cases}\\
h_{t+1}&=f(h_t,u_t,N_t):=2(H_0+\rho_t^2)-h_t.
\end{aligned}
$$

This closed-form mapping eliminates iterative root finding. The total release $r_t$ is recovered from the rating relation; downstream stages are computed by the existing St. Lawrence routing operator, $d_t=\mathcal R(\bar h_t,r_t;\xi_t)$, with $\xi_t$ denoting other hydrologic and operational forcings.

### Reference rule curve

The legacy baseline uses a long-term supply forecast $F_t$ and a four-quarter short-term forecast $(\widehat N_{t+j\mid t})_{j=0}^{3}$. Writing $\bar h_t^{(48)}=48^{-1}\sum_{i=1}^{48}h_{t-i}^{\mathrm{MOQ}}$, its additive adjustment (in units of $10\,\mathrm{m}^3\,\mathrm{s}^{-1}$) is

$$
\psi_t=
\begin{cases}
\widetilde C_{1,t}\left(\dfrac{F_t-7011}{1541}\right)^{P_1},
&F_t\ge7011,\\[6pt]
-C_2\left(\dfrac{7011-F_t}{1294}\right)^{P_2},
&F_t<7011.
\end{cases}
$$

Here $\widetilde C_{1,t}\in\{C_1,C_1^*\}$ is selected by the wet-confidence test. The baseline constructs four forecast releases $r^{\mathrm{RC}}_{t,j}$ from the rating relation plus $\psi_t$, including release rounding and any dry-level correction triggered by $\bar h_t^{(48)}$, and applies

$$
r_t^{\mathrm{RC}}=\operatorname{round}\!\left(
\tfrac14\sum_{j=0}^{3}r^{\mathrm{RC}}_{t,j}\right).
$$

Thus, the four-quarter computation is a *policy forecast*, while the environmental simulation advances one quarter-month at a time.

## Finite-horizon Bolza optimal control

Let $x_t$ contain $h_t$, relevant downstream and lagged operational variables, and any auxiliary history $z_t$ required by terminal performance indicators. In continuous time, the Bolza objective is

$$
\Phi(x(T))+\int_{t_0}^{T}\mathcal L(x(\tau),u(\tau);\xi(\tau))\,d\tau.
$$

For a horizon $H$ measured in quarter-months, its discrete-time counterpart is

$$
\begin{aligned}
\underset{u_{t:t+H-1}}{\operatorname{minimize}}\quad
&J_t=\sum_{j=0}^{H-1}
\ell(x_{t+j},u_{t+j};\xi_{t+j\mid t})
+\Phi(x_{t+H}),\\[5pt]
\text{subject to}\quad
&x_{t+j+1}=\mathcal F(x_{t+j},u_{t+j};\xi_{t+j\mid t}),
&&j=0,\ldots,H-1,\\
&r_{t+j}\in\mathcal Q_{\mathrm{Bv7}}(x_{t+j},\xi_{t+j\mid t}),
&&j=0,\ldots,H-1,\\
&u_{\min}\le u_{t+j}\le u_{\max},\qquad
\bar h_{t+j}\ge H_0,
&&j=0,\ldots,H-1,\\
&x_t=x_t^{\mathrm{observed}}.
\end{aligned}
$$

The transition $\mathcal F$ contains the Lake Ontario closed-form mapping, the total-release decoder, and St. Lawrence routing. The set $\mathcal Q_{\mathrm{Bv7}}$ represents the applicable Plan 2014 operational release limits.

For the current four-indicator MPC implementation, the running cost is

$$
\boxed{
\ell_t=w_{\mathrm{UC}}C_{\mathrm{UC},t}
+w_{\mathrm{DC}}C_{\mathrm{DC},t}
-w_{\mathrm{HP}}V_{\mathrm{HP},t}
+w_{\mathrm{RB}}C_{\mathrm{RB},t}.
}
$$

Here UC/DC are upstream/downstream coastal impacts, HP is hydropower value, and RB is recreational-boating impact. The current MPC sets $\Phi\equiv0$ and uses baseline-normalized weights. An extended formulation can include commercial navigation in $\ell_t$ and meadow-marsh and muskrat indicators in $\Phi(x_{t+H})$ after augmenting the state with the history required to evaluate these ecological metrics.

In **single direct shooting**, all predicted states are eliminated by recursion through $\mathcal F$, leaving $u_{t:t+H-1}$ as the decision vector. Only $u_t^*$ is applied; optimization is repeated at $t+1$. The present numerical implementation enforces Plan 2014 limits through a post-decision release correction, rather than by explicitly imposing all the constraints in the Bolza problem on the optimizer.

## Comparison of reservoir transition solvers

We evaluate three methods for advancing the Lake Ontario stage over one quarter-month, using identical $h_t$, net total supply $N_t$, and discharge adjustment $u_t$. Let $\alpha=1/c=\Delta t/A_{\mathrm{ON}}$ and $Q(h,u)=A_w(h-H_0)^{3/2}+10u$.

**Forward Euler (explicit).** Evaluating the discharge at the beginning of the interval gives

$$
h_{t+1}^{\mathrm{FE}}=h_t+\alpha\left[N_t-Q(h_t,u_t)\right].
$$

**Implicit midpoint (Newton iteration).** Evaluating the discharge at the mean stage requires solving $F_t(z)=0$, where

$$
\begin{aligned}
F_t(z)&=z-h_t-\alpha\left[N_t-Q\!\left(\frac{h_t+z}{2},u_t\right)\right],\\
F_t'(z)&=1+\frac{3\alpha A_w}{4}\sqrt{\frac{h_t+z}{2}-H_0},\\
z^{(k+1)}&=z^{(k)}-\frac{F_t(z^{(k)})}{F_t'(z^{(k)})},
\qquad z^{(0)}=h_{t+1}^{\mathrm{FE}}.
\end{aligned}
$$

Upon convergence, $h_{t+1}^{\mathrm{IM}}=z^*$.

**Closed-form cubic (analytical midpoint).** Substituting $\rho_t=\sqrt{(h_t+h_{t+1})/2-H_0}$ reduces the same midpoint equation to $\rho_t^3+a\rho_t^2-b_t=0$, with $a$ and $b_t$ defined above. Its admissible nonnegative root gives

$$
h_{t+1}^{\mathrm{CF}}=2\left(H_0+\rho_t^2\right)-h_t,
\qquad \rho_t=\frac{a}{3}\left[2\chi\!\left(\frac{27b_t}{2a^3}-1\right)-1\right].
$$

Forward Euler is a first-order time discretization; implicit midpoint is second-order under smooth forcings. **Newton iteration and the closed-form cubic solve the same implicit midpoint equation**, so they should agree to numerical tolerance. The closed form avoids iterative root finding; comparisons of stage trajectories, midpoint residuals, and computational cost can assess their practical differences. All three methods are evaluated on the admissible hydraulic-head domain.


![alt text](image-1.png)
![alt text](image-2.png)
![alt text](image-3.png)


