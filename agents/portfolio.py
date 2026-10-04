"""Portfolio-construction agent.

Each month it runs the same loop: validate inputs -> drift last month's book ->
combine signals and devil's-advocate verdicts into one alpha -> solve -> verify
with the independent guardrails -> relax soft limits one step and retry if
needed -> explain every trade. Decisions are deterministic: no LLM is called and
no forward return is ever loaded here, so there is no model-side look-ahead.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd

from portfolio_agent.config import RELAX_LADDER, Mandate, Policy
from portfolio_agent.contracts import ContractError, validate_da, validate_signals
from portfolio_agent.data import OUTCOME_COL
from portfolio_agent.guardrails import MandateViolation, check_portfolio, enforce, hard_failures
from portfolio_agent.optimizer import EPS, solve


@dataclass
class Rebalance:
    eom: pd.Timestamp
    weights: pd.Series        # post-trade, non-zero names
    pre_trade: pd.Series      # drifted weights of names still in the panel
    trades: pd.DataFrame
    checks: list
    diagnostics: dict


@dataclass
class RunResult:
    holdings: pd.DataFrame    # submission format + eom, label_status
    trades: pd.DataFrame
    monthly: pd.DataFrame     # one row of diagnostics per rebalance
    checks: pd.DataFrame      # every guardrail check, every month


class PortfolioConstructionAgent:
    def __init__(self, panel: pd.DataFrame, mandate: Mandate | None = None,
                 policy: Policy | None = None):
        if OUTCOME_COL in panel.columns:
            raise ContractError(f"panel passed to the agent must not contain {OUTCOME_COL}")
        self.mandate = mandate or Mandate()
        self.policy = policy or Policy()
        self.policy.validate(self.mandate)
        self.by_month = {t: g.set_index("permno") for t, g in panel.groupby("eom")}

    # ------------------------------------------------------------------ alpha
    def _book(self, rows: pd.DataFrame, sig: pd.DataFrame, da: pd.DataFrame) -> pd.DataFrame:
        p = self.policy
        book = pd.DataFrame(index=rows.index)
        book["sector"] = rows["sector"]
        liquid = (rows["prc"] >= p.min_price) & (rows["dolvol_126d"] >= p.min_dolvol)
        book["elig"] = liquid & rows["size_grp"].isin(p.size_groups)
        book["elig_short"] = (book["elig"] & rows["size_grp"].isin(p.short_size_groups)
                              & (rows["dolvol_126d"] >= p.short_min_dolvol))
        beta = rows["beta_60m"]
        book["beta_imputed"] = beta.isna()
        med = beta[book["elig"]].median()
        book["beta"] = beta.fillna(1.0 if np.isnan(med) else med)

        universe = book.index[book["elig"]]
        sig = sig[sig["permno"].isin(universe)]
        total, weight_sum = pd.Series(0.0, index=universe), 0.0
        scored = pd.Series(False, index=universe)
        self._sources = sorted(sig["source"].unique())
        for src in self._sources:
            s = sig[sig["source"] == src].set_index("permno")
            sd = s["score"].std(ddof=0)
            z = (s["score"] - s["score"].mean()) / sd if sd > 0 else s["score"] * 0.0
            z = (z.clip(-p.winsor_z, p.winsor_z) * s["confidence"]).reindex(universe)
            book["z_" + src] = z
            wt = p.source_weights.get(src, 1.0)
            total += wt * z.fillna(0.0)      # no view from a source = neutral
            weight_sum += wt
            scored |= z.notna()
        alpha = (total / weight_sum).where(scored) if weight_sum else total.where(scored)
        if p.sector_neutral_alpha:
            alpha = alpha - alpha.groupby(book.loc[universe, "sector"]).transform("mean")
        sd = alpha.std(ddof=0)
        book["alpha_raw"] = alpha / sd if sd and sd > 0 else alpha
        notes = sig[sig["rationale"] != ""].groupby("permno")["rationale"].agg(" | ".join)
        book["signal_note"] = notes.reindex(book.index).fillna("").str.slice(0, 300)

        da = da.set_index("permno").reindex(book.index)
        book["da_action"] = da["action"].fillna("pass")
        book["da_reason"] = da["reason"].fillna("")
        side, veto = da["side"].fillna("both"), book["da_action"] == "veto"
        book["veto_long"] = veto & side.isin(["long", "both"])
        book["veto_short"] = veto & side.isin(["short", "both"])
        raw = book["alpha_raw"]
        hit = (book["da_action"] == "haircut") & (
            (side == "both") | ((side == "long") & (raw > 0)) | ((side == "short") & (raw < 0)))
        severity = pd.to_numeric(da["severity"]).fillna(0.0).astype(float)
        book["cap_mult"] = np.where(hit, 1.0 - severity, 1.0)
        book["alpha"] = raw * book["cap_mult"]
        book["rank_long"] = book["alpha"].rank(ascending=False)
        book["rank_short"] = book["alpha"].rank(ascending=True)
        return book

    # -------------------------------------------------------------- rebalance
    def rebalance(self, eom, signals_t: pd.DataFrame, da_t: pd.DataFrame,
                  prev: pd.Series | None = None) -> Rebalance:
        p, m = self.policy, self.mandate
        eom = pd.Timestamp(eom)
        rows = self.by_month[eom]
        book = self._book(rows, signals_t, da_t)
        first = prev is None or len(prev) == 0

        # Drift last month's weights with the return just realised (month t).
        if first:
            w0, gone = pd.Series(dtype=float), pd.Series(dtype=float)
        else:
            here = prev.index.intersection(rows.index)
            gone = prev.drop(here)                      # delisted / left the panel
            rf = float((rows["ret"] - rows["ret_exc"]).median())
            exc = rows["ret_exc"].reindex(prev.index).fillna(p.missing_return_fill)
            port_ret = rf + float((prev * exc).sum())
            w0 = prev[here] * (1.0 + rows.loc[here, "ret"].fillna(0.0)) / (1.0 + port_ret)
        forced = float(gone.abs().sum())

        cap = p.max_weight * book["cap_mult"]
        alpha = book["alpha"]
        held = w0.reindex(book.index).fillna(0.0)
        ub_long = np.where(book["elig"] & (alpha > 0), cap, 0.0)
        ub_long = np.where(held > 0, np.where(book["elig"], cap, np.minimum(cap, held)), ub_long)
        ub_long = np.where(book["veto_long"], 0.0, ub_long)
        ub_short = np.where(book["elig_short"] & (alpha < 0), cap, 0.0)
        # A short that fails the borrow/liquidity screen must be covered, not grandfathered.
        ub_short = np.where(held < 0, np.where(book["elig_short"], cap, 0.0), ub_short)
        ub_short = np.where(book["veto_short"], 0.0, ub_short)

        use = (ub_long > 0) | (ub_short > 0) | (held.values != 0)
        idx = book.index[use]
        args = (alpha[idx].fillna(0.0).values, held[idx].values, ub_long[use], ub_short[use],
                book.loc[idx, "beta"].values, book.loc[idx, "sector"].values)

        chosen = None
        for level, relax in enumerate(RELAX_LADDER):
            to_mult = relax.get("turnover", 1.0)
            budget = None if (first or to_mult is None) else 2 * p.max_turnover * to_mult - forced
            x = solve(*args, gross_max=p.gross_target,
                      net_lo=p.net_target - p.net_band, net_hi=p.net_target + p.net_band,
                      beta_tol=p.beta_tol * relax.get("beta", 1.0),
                      sector_tol=p.sector_tol * relax.get("sector", 1.0),
                      turnover_two_way=budget, trade_penalty=p.trade_penalty)
            if x is None:
                continue
            w = pd.Series(x, index=idx)
            w = w[w != 0]
            turnover = None if first else 0.5 * (
                float(w.reindex(idx).fillna(0.0).sub(held[idx]).abs().sum()) + forced)
            checks = check_portfolio(w, m, p, beta=book["beta"], sector=book["sector"],
                                     turnover=turnover, short_eligible=book["elig_short"])
            if not hard_failures(checks):
                chosen = (level, relax, w, turnover, checks)
                break
        if chosen is None:
            raise MandateViolation(
                f"{eom.date()}: no portfolio satisfies the mandate even with soft limits "
                f"relaxed ({int(book['elig'].sum())} eligible names, "
                f"{int(book['alpha'].notna().sum())} scored)")
        level, relax, w, turnover, checks = chosen
        enforce(checks, str(eom.date()))

        diagnostics = {
            "eom": eom, "date": eom + pd.Timedelta(days=1),
            "n_long": int((w > 0).sum()), "n_short": int((w < 0).sum()),
            "gross": float(w.abs().sum()), "net": float(w.sum()),
            "long_exposure": float(w[w > 0].sum()), "short_exposure": float(w[w < 0].sum()),
            "max_weight": float(w.abs().max()),
            "top10_share": float(w.abs().nlargest(10).sum() / w.abs().sum()),
            "turnover_one_way": turnover,
            "forced_exit_weight": forced,
            "relax_level": level,
            "relaxations": ", ".join(f"{k} x{v}" if v else f"{k} off" for k, v in relax.items()),
            "policy_breaches": ", ".join(c.name for c in checks if not c.ok),
            "universe": int(book["elig"].sum()), "scored": int(book["alpha"].notna().sum()),
            "vetoes": int((book["da_action"] == "veto").sum()),
            "haircuts": int((book["da_action"] == "haircut").sum()),
            "beta_imputed_weight": float(w[book.loc[w.index, "beta_imputed"]].abs().sum()),
        }
        for c in checks:
            diagnostics["chk_" + c.name] = c.value
        trades = self._explain(eom, book, held, w, gone, rows)
        return Rebalance(eom, w, w0, trades, checks, diagnostics)

    # ------------------------------------------------------------ explanation
    def _explain(self, eom, book, held, w, gone, rows) -> pd.DataFrame:
        """One row per trade with the reason it was made."""
        p = self.policy
        new = w.reindex(book.index).fillna(0.0)
        moved = book.index[(new - held).abs() > EPS]
        n_scored = int(book["alpha"].notna().sum())
        out = []
        for permno in moved:
            b, old, tgt = book.loc[permno], held[permno], new[permno]
            if old == 0:
                action = "OPEN_LONG" if tgt > 0 else "OPEN_SHORT"
            elif tgt == 0:
                action = "CLOSE_LONG" if old > 0 else "COVER_SHORT"
            elif np.sign(old) != np.sign(tgt):
                action = "FLIP"
            else:
                action = "ADD" if abs(tgt) > abs(old) else "TRIM"
            a = b["alpha"]
            view = ("no score this month" if pd.isna(a) else
                    f"alpha z={a:+.2f} (long rank {int(b['rank_long'])}, "
                    f"short rank {int(b['rank_short'])} of {n_scored})")
            parts = ", ".join(f"{s}={b['z_' + s]:+.2f}" for s in self._sources
                              if pd.notna(b["z_" + s]))
            cap = p.max_weight * b["cap_mult"]
            long_side = old > 0 or (old == 0 and tgt > 0)
            vetoed = b["veto_long"] if long_side else b["veto_short"]
            eligible = b["elig"] if long_side else b["elig_short"]
            if action.startswith("OPEN") or action in ("ADD", "FLIP"):
                why = f"{view}; sources: {parts or 'n/a'}; devil's advocate: {b['da_action']}"
                if b["da_action"] == "haircut":
                    why += f" (size cap cut to {cap:.2%}: {b['da_reason']})"
            elif vetoed:
                why = f"devil's-advocate veto: {b['da_reason'] or 'no reason given'}"
            elif not eligible:
                why = "no longer passes the size / price / liquidity screen for this side"
            elif abs(old) > cap + EPS and abs(tgt) >= cap - EPS:
                why = f"drifted above the {cap:.2%} position cap"
            elif tgt == 0 or abs(old - tgt) > 0.25 * p.max_weight:
                why = f"{view}; capital rotated to higher-conviction names"
            else:
                why = "small resize to restore dollar / beta / sector neutrality"
            out.append((permno, rows.at[permno, "ticker_ff"], rows.at[permno, "company_name_ff"],
                        action, old, tgt, a, why, b["signal_note"]))
        last = self.by_month.get(eom - MonthEnd(1))
        for permno, old in gone.items():
            known = last is not None and permno in last.index
            out.append((permno, last.at[permno, "ticker_ff"] if known else None,
                        last.at[permno, "company_name_ff"] if known else None,
                        "CLOSE_LONG" if old > 0 else "COVER_SHORT", old, 0.0, np.nan,
                        "no longer in the panel (delisted, acquired or dropped); forced exit", ""))
        t = pd.DataFrame(out, columns=["permno", "ticker", "company_name", "action",
                                       "weight_before", "weight_after", "alpha", "reason",
                                       "signal_note"])
        t.insert(0, "date", eom + pd.Timedelta(days=1))
        return t

    # -------------------------------------------------------------------- run
    def run(self, signals: pd.DataFrame, da: pd.DataFrame | None = None,
            start="2020-12-31", end="2026-07-31") -> RunResult:
        """start / end are formation month-ends; holdings are dated the next day."""
        signals, da = validate_signals(signals), validate_da(da)
        months = [t for t in sorted(self.by_month) if pd.Timestamp(start) <= t <= pd.Timestamp(end)]
        sig_by, da_by = dict(tuple(signals.groupby("eom"))), dict(tuple(da.groupby("eom")))
        prev, holdings, trades, monthly, checks = None, [], [], [], []
        for t in months:
            if t not in sig_by:
                raise ContractError(f"no signals supplied for formation month {t.date()}")
            r = self.rebalance(t, sig_by[t], da_by.get(t, da.iloc[:0]), prev)
            prev = r.weights
            rows = self.by_month[t].loc[r.weights.index]
            h = pd.DataFrame({
                "Date": t + pd.Timedelta(days=1), "PERMNO": r.weights.index,
                "TICKER": rows["ticker_ff"].values, "COMPANY NAME": rows["company_name_ff"].values,
                "WEIGHT": r.weights.values, "eom": t,
                "label_status": np.where(rows["ticker"].notna(), "verified_in_month",
                                         np.where(rows["ticker_ff"].notna(),
                                                  "carried_forward", "unlabeled")),
            })
            holdings.append(h.sort_values("WEIGHT", ascending=False))
            trades.append(r.trades)
            monthly.append(r.diagnostics)
            checks += [{"date": r.diagnostics["date"], **c.as_dict()} for c in r.checks]
        return RunResult(pd.concat(holdings, ignore_index=True),
                         pd.concat(trades, ignore_index=True),
                         pd.DataFrame(monthly), pd.DataFrame(checks))
