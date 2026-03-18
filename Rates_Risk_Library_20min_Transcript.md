# Rates Risk Library — 20-minute talk transcript (slide-by-slide)

## Slide 1 — Title
Hi everyone, I’m Joshua Cheng. Today I’ll walk through my Rates Risk Library: a desk-style analytics stack that starts from market data and ends with an interactive dashboard for pricing and risk integrated NSS and SABR model, which help in pricing, hedging risk, scenarios, VaR revaluation, and P&L explain.

## Slide 2 — Agenda
I’ll start with the desk problem we’re trying to solve, then the library architecture and the main data contracts.
After that, I’ll cover the core analytics: curve construction, linear pricing, risk metrics, VaR/ES, and P&L explain.
Then I’ll slow down for the SABR part — especially the hedging logic I learn from the paper.
Finally, I’ll show how everything is exposed in the Streamlit dashboard, tab by tab.

## Slide 3 — Motivation: Desk-Style Consistency
So why do we need a library for risk management?
On a trading desk, inconsistent numbers are the fastest way to lose trust in a risk report.
If curve building, pricing, and vol live in separate places, you’ll inevitably get mismatched price and noisy explanations.
Also, automating the risk management workflow is no longer a “nice-to-have”; it’s essential for modern financial institutions. The core reasons fall into speed, accuracy, scalability, governance, and decision quality.
So the goal here is one consistent source of truth — with explicit conventions and explicit trade definitions — and then a UI that makes those results easy to inspect.

## Slide 4 — What the Library Delivers
Let me summarize the deliverables in two buckets.
On the analytics side: curves, pricing for common rates instruments, risk metrics like DV01 and key-rate DV01, P&L explain, and VaR/ES with backtesting.
On the production side: full SABR integration, failure tracking so we never silently ignore a trade, and a MarketState container so runs are reproducible and fast.
And then everything is connected to a dashboard so a user can navigate the workflow without reading the code.

## Slide 5 — End-to-End Flow
4 layer
let me work you through it
Market data flows into two state objects: CurveState for discounting/forwards, and VolState for the SABR surface.
Those feed the engines — pricing and risk — and the dashboard is simply a UI layer on top of the same objects.
So I’m not “reimplementing” logic in the UI. The UI calls the same library functions the batch scripts would call.

## Slide 6 — Module Map
Core modules are classes in charge of conventions, dates, curves, pricers, and risk.
Then there’s a dedicated vol/options layer for SABR and caplet/swaption logic.
On top, the risk stack modules include VaR, P&L, liquidity, and reporting.
Finally, portfolio, market_state, and dashboard are the workflow and UI glue that connects everything.

## Slide 7 — Data Contracts: MarketState + Trade Builders
A big “make it work in practice” decision is to use explicit data contracts.
MarketState holds the curves, the SABR surface, and portfolio inputs in one place, and it’s what every tab reads from.
Positions are converted into explicit trades using builder functions — bond, swap, caplet, swaption.
Namely, 
• It accepts a pos (a Pandas Series representing a single row of data), a valuation_date, and the market_state.
• It reads the instrument_type field from the data and routes the request to the specific builder function required (e.g., if type is "SWAP", it calls build_swap_trade).
• It wraps the construction process in error handling blocks to catch specific data issues, such as PositionValidationError or MissingFieldError
and Output A dictionary containing, for example, the standard bond definition used for calculating Dirty/Clean prices and Accrued Interest.

The point is: nothing is guessed. If a field is missing or inconsistent, we raise a clear error and surface it in diagnostics.

## Slide 8 — Curve Foundations
Here are the curve basics the engine uses: discount factors, zero rates, and the OIS bootstrap step.
OIS bootstrapping is basically solve the next discount factor so the next instrument reprices exactly.
For Treasuries, I use Nelson–Siegel–Svensson because it’s smooth and interpretable — you can explain level, slope, and curvature moves to a risk manager. Also, it is consistent with the idea of PCA decomposition.

## Slide 9 — Linear Pricing
Linear pricing is intentionally curve-driven.
Bonds are discounted cashflows. Swaps are fixed leg PV versus floating leg PV.
SOFR futures are quoted as 100 minus an implied rate, and that implied rate is linked to discount factors.
So if the curve is right, linear pricing is consistent by construction — and then risk comes from bumping the same curve object.

## Slide 10 — DV01 and Key-Rate DV01
This slide is the core of the risk engine: bump-and-reprice.
DV01 is the central difference sensitivity to a 1bp parallel shift.
Key-rate DV01 bumps one tenor at a time to capture twists and butterflies.
The practical output is that the dashboard can show both total DV01 and where it sits along the curve — and even highlight the “worst key rate” exposure.

## Slide 11 — VaR/ES + Backtesting
VaR is computed by full revaluation: build a shocked curve, reprice the whole book, and collect P&L.
That naturally captures convexity and option nonlinearities.
Expected shortfall is the average of losses beyond VaR, so it’s a better tail-risk measure.
And backtesting is included because it’s how you check whether your VaR model is underestimating risk in practice.

## Slide 12 — P&L Explain
Here’s the explain framework: carry, roll-down, curve move, convexity, and then residual.
The curve move term is predicted directly from key-rate DV01 times realized key-rate changes.
For option books, I extend this by separating curve-driven changes from vol-driven changes — and that’s where SABR integration becomes important.

## Slide 13 — Why SABR?
This slide is the motivation from the memo.
If you try to fit option prices exactly, local volatility gives you a surface, but it doesn’t guarantee the right dynamics for hedging.
SABR is a practical stochastic-vol model that produces a surface and, more importantly, a consistent way to map underlying moves into smile moves.

## Slide 14 — What “Full SABR Integration” Means
“Full integration” means the SABR surface is used everywhere.
We ingest normal or lognormal quotes, with optional shifting.
We calibrate parameters per expiry/tenor, then use the SABR-implied vol for pricing caplets and swaptions.
And critically, we compute SABR-consistent hedge sensitivities and feed those into scenarios, VaR revaluation, and P&L explain.
So vol isn’t an add-on — it’s a first-class risk factor.

## Slide 15 — SABR Model (Shifted)
This is the shifted SABR dynamics.
The shift lambda is crucial in rates markets because forwards can be low or even negative.
Conceptually: beta controls how volatility scales with the level of the forward, rho controls skew, and nu controls smile curvature.
Once we have those parameters, we map them to an implied vol under either Black or Bachelier, depending on the market quote convention.

## Slide 16 — Calibration and Implicit Differentiation
Instead of treating alpha as free, desks often parameterize with ATM volatility: sigma_ATM plus beta, rho, and nu.
Alpha becomes an implied function of sigma_ATM and the other parameters.
Then implicit differentiation gives you stable derivatives like d alpha / d sigma_ATM, which is exactly what we want for ATM-vega and fast intraday updates.

## Slide 17 — Model-Consistent Delta (Sideways vs Backbone)
This is the main hedging slide.
SABR delta is Black delta plus a correction: Black vega times how implied vol changes with the forward.
And that implied-vol change can be decomposed into sideways motion — sticky log-moneyness — and the backbone, which comes from ATM volatility moving with the forward.
It agains shows the drawback of flat volatility and local volatility

## Slide 18 — ATM Vega in Desk Coordinates
Here we report vega with respect to sigma_ATM, not just with respect to a model parameter like alpha.
That matters because traders quote and risk-manage in ATM vols.
The chain rule connects vega to sigma_ATM through implied vol, then through the SABR mapping, then through alpha as a function of sigma_ATM.

## Slide 19 — Dashboard Structure
Now we move from the library into the user experience.
The dashboard is Streamlit, and it’s intentionally structured as a guided workflow.
There are eight tabs: Curves, Pricing, Risk Metrics, VaR, Scenarios, P&L Explain, Liquidity, and Data Explorer.
The key idea is that each tab reads from the same MarketState object, so the numbers remain consistent across views.

## Slide 20 — Tab 1: Curves
In Curves, the user can see the OIS and Treasury/NSS curves, along with discount factors and forwards.

This tab also contains the SABR surface visualization, including residual buckets and fitted smile curves, which is essential for sanity-checking calibration quality.

## Slide 21 — Tab 2: Pricing
Pricing is where we verify that the market state produces reasonable PVs.
The tab shows PV per instrument and also breakdown views — what cashflows exist, how discounting works, and which model is used for option products.

## Slide 22 — Tab 3: Risk Metrics
Risk Metrics is the day-to-day desk view.
It reports DV01, key-rate DV01, and convexity for linear products.
For options, it reports SABR-consistent delta and ATM-vega, along with diagnostics.
And it supports drill-down — for example, which key rate is the largest driver of risk.

## Slide 23 — Tab 4: VaR
VaR runs historical simulation and Monte Carlo.
It reports VaR and ES at common confidence levels, and it supports horizon choices.
<!-- The key operational piece is backtesting — showing exceedances so you can evaluate whether the model is too optimistic. -->

## Slide 24 — Tab 5: Scenarios
Scenarios are for stress testing and for trader “what-if” questions.
There are standard scenarios like parallel up/down, steepeners, flatteners, and twists.
There’s also a custom scenario builder where the user specifies bumps at key tenors.
For option books, the scenario view can also separate curve impact versus vol impact.

## Slide 25 — Tab 6: P&L Explain
This tab turns risk into explanation.
It decomposes realized P&L into carry, roll-down, curve move, convexity, and residual.
The curve move term is driven by the actual key-rate changes, applied to yesterday’s key-rate DV01.
For options, it’s also the natural place to add smile and vol explain, because SABR gives you a consistent structure.

## Slide 26 — Tab 7: Liquidity
Liquidity is a lightweight but important extension.
It highlights positions that may be harder to exit — for example, wider bid/ask or lower assumed liquidity buckets.
Even if the initial version is simple, the goal is to provide a natural place to extend into liquidity-adjusted risk metrics.

## Slide 27 — Tab 8: Data Explorer
Data Explorer is essentially the audit and debugging layer.
It lets you inspect raw inputs — curve instruments, vol quotes, positions — and the normalized versions used by the engine.
When something looks off, this tab answers the question: “what data did we actually use to produce today’s risk?”

## Slide 28 — Validation
Before trusting results, I validate at multiple levels.
Curves should reprice their input instruments within tight tolerance.
DV01 should be stable under central differences.
SABR calibration should produce reasonable parameters and sensible residual plots.
And at the end-to-end level, yesterday-to-today changes should be explainable with a bounded residual.

## Slide 29 — Roadmap
Finally, some natural next steps.
Multi-curve support and richer futures convexity adjustments.
Speeding up VaR with factor models and incremental revaluation.
And longer-term: XVA hooks and FRTB-style risk factor mapping.
But the core foundation — consistent market state, pricing, risk, and SABR — is already in place.

## Slide 30 — Conclusion
To wrap up: the project is a unified desk-style stack.
It starts from market data and ends with an explainable risk report in the dashboard.
The main differentiator is that SABR is integrated into pricing and hedging, not treated as an isolated model.
And the dashboard makes the whole system operational for daily use.
Thanks — and I’m happy to walk through a live demo or answer questions.
