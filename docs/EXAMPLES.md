# Sample outputs & evaluation artifacts

Real captured output from this project's own test runs — nothing here is
invented or idealized. Where a run failed (the CrewAI backend flakiness
documented in the README's audit trail #5), that's shown too, since it's
part of the honest picture of how the system behaves.

## Deterministic evaluator runs

`python -m market_forecaster.evaluators`

```
=== Router classification ===
[PASS] expected='straight' actual='straight' question='What is AAPL trading at right now?'
[PASS] expected='straight' actual='straight' question="What's TSLA's dividend yield?"
[PASS] expected='straight' actual='straight' question="What is NVDA's trailing PE ratio?"
[PASS] expected='straight' actual='straight' question="What is MSFT's analyst rating?"
[PASS] expected='tot' actual='tot' question='Why did MRNA stock go down?'
[PASS] expected='tot' actual='tot' question='Should I rebalance given my sector concentration?'
[PASS] expected='tot' actual='tot' question='What do you think about my exposure to tech right now?'
[PASS] expected='tot' actual='tot' question='Should I be worried about a downturn hitting my portfolio?'
Accuracy: 100% (8/8)

=== ReAct tool selection ===
[PASS] expected='get_holding_data' actual='get_holding_data' args={'ticker': 'AAPL'} question='What is AAPL trading at right now?'
[PASS] expected='get_holding_data' actual='get_holding_data' args={'ticker': 'MSFT'} question="What's MSFT's dividend yield?"
[PASS] expected='search_filings' actual='search_filings' args={'ticker': 'NVDA', 'query': 'risk factors business risks'} question="What does NVDA's 10-K say about its business risk factors?"
[PASS] expected='search_client_history' actual='search_client_history' args={'query': 'AAPL discussion'} question='What did we discuss about AAPL the last time we talked?'
Accuracy: 100% (4/4)
```

## Calibration run — an honest negative result

`python -m market_forecaster.evaluators --calibration`

```
ERROR:crewai.flow.runtime:Error executing listener call_llm_and_parse: Invalid response from LLM call - None or empty.
WARNING:__main__:CalibrationEvaluator: tot_pipeline failed for question='Should I rebalance given my concentration in tech?', skipping
WARNING:market_forecaster.guardrails:critic_output_guardrail: critic omitted the CONFIDENCE line, ConfidenceRouter will have no score for this turn
WARNING:__main__:CalibrationEvaluator: tot_pipeline failed for question='Should I be worried about a pullback given how richly valued these are?', skipping

=== Confidence calibration (ToT critic score vs. relevancy) ===
confidence=unknown  (score=None) relevancy_pass_rate=0.94  question='What do you think about my exposure to AI-driven growth right now?'

Average relevancy pass_rate by confidence bucket:
  unknown  0.94
```

**What this shows**: 2 of 3 cases failed outright on the backend's
documented intermittent empty-response issue, and the one that succeeded
had a critic that skipped its own required `CONFIDENCE:` line. The harness
itself worked correctly — it skipped failures without crashing, and
correctly bucketed the missing score as `unknown` rather than guessing —
but one data point in an `unknown` bucket says nothing about whether "low
confidence" tracks real answer quality. See the README's Known Limitations
for what a real calibration signal would require.

## Sample conversation: portfolio submission

Input: `AAPL,10 MSFT,5`

Output:
> Got it — here's your profile:
>
> This portfolio is heavily concentrated in just two mega-cap Technology
> names, AAPL and MSFT, with no diversification across sectors—both sit in
> overlapping Tech/software-hardware exposure, so any tech-specific
> drawdown would hit the entire book. Dividend income is minimal (yields
> of 0.34% and 0.71%), making this a growth-oriented rather than
> income-generating allocation. Valuations are elevated: AAPL trades at
> 36.6x trailing/33.5x forward earnings, and MSFT at 28.6x trailing/21.8x
> forward, both pricing in continued strong execution...
>
> Ask me anything about it.

## Sample conversation: ReAct fact lookup

Input: `What is AAPL trading at right now?`

Output:
> AAPL is currently trading at **$319.70**, up about 39.2% over the past
> year and 2.5% over the last 3 months (range: $273.51–$344.27). Analysts
> rate it a "buy" with a mean target price of $324.45, suggesting modest
> further upside expected. It also beat earnings expectations last quarter
> (EPS of $2.02 vs. $1.89 estimate, a 6.74% surprise).

## Sample conversation: ToT deliberation with source citations

Input: `Why did AAPL move recently?` (routed to `tot`)

Output (excerpt):
> ## Why AAPL Moved
> The recent move is a **sentiment- and narrative-driven rally**, not a
> fundamentals repricing. Two forces converged:
>
> 1. **Leadership succession headlines** — John Ternus's positioning as
>    Apple's next CEO is being framed as continuity into "the age of AI,"
>    giving the market a constructive story to trade on (Source: "John
>    Ternus to lead Apple into the age of AI", 2026-08-31).
> 2. **Fundamental cover from the last print** — a solid but unspectacular
>    6.7% earnings beat gives the rally a veneer of fundamental
>    justification, even though it predates and doesn't fully explain the
>    move's magnitude (Source: Earnings, 2026-07-30).

Both citations trace to real fields already fetched for the profile
summary (`raw_data[ticker]["recent_news"]` / `["earnings"]`) — nothing is
invented; see `agents/tot_crew.py::_format_citable_sources`.

## Sample: portfolio rollup alert firing on a real diff

Simulated a prior check-in where AAPL's rating and price differed, then
resubmitted the same portfolio through the actual `respond()` pipeline:

```
📋 1 of your 2 holdings had changes since your last check-in, 1 high-impact:
- AAPL: analyst rating changed from sell to buy

Got it — here's your profile:
...
```

MSFT (unchanged) was correctly excluded — see `core/alerts.py`.

## Sample: guardrails firing correctly

`ContentPolicyGuard` on a deliberately advice-framed sentence:

```
check("You should buy TSLA right now.")
-> flagged=True, matched_phrases=['buy']

annotate(...) ->
"[This response was flagged for advice-like phrasing -- treat it as
factual/explanatory context only, not a personal investment
recommendation.]

You should buy TSLA right now."
```

`ActionConstraintGuard`'s startup audit, from a real app boot:

```
INFO market_forecaster.guardrails: ActionConstraintGuard: startup audit passed, 7 agents checked, all read-only
```
