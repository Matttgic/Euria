"""Routine quotidienne (GitHub Actions) : règle les paris, envoie le bilan, cherche les value bets.

Ne plante jamais sur une panne de source : chaque donnée manquante est signalée dans les logs
et, si besoin, dans le message Telegram."""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timedelta, timezone

from . import analysis, betting, config, services, telegram
from .config import LEAGUES
from .stats import FEATURE_NAMES

log = logging.getLogger("euria.bot")

PREDICTION_COLUMNS = [
    "run_at", "league", "kickoff", "home", "away", "p_home", "p_draw", "p_away", *FEATURE_NAMES,
    "home_rank", "away_rank", "temperature_c", "precipitation_mm", "wind_speed_ms",
    "home_luck", "away_luck", "second_opinion_outcome", "second_opinion_prob",
    "bookmaker", "value_bet", "matches_source", "odds_source", "weather_source",
]
OUTCOME_FR = {"Home": "Domicile", "Draw": "Nul", "Away": "Extérieur"}


def _log_prediction(row: dict) -> None:
    """Journal des prédictions pour un futur réentraînement. Volontairement sans les cotes brutes
    (conditions Parlay : pas de republication ni de conservation > 90 jours, et le dépôt est public)."""
    path = config.PREDICTIONS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open(encoding="utf-8") as f:
            header = f.readline().strip().split(",")
        if header != PREDICTION_COLUMNS:  # colonnes changées : on archive l'ancien journal au lieu de le mélanger
            path.rename(path.with_name(f"{path.stem}_{datetime.now(timezone.utc):%Y%m%d%H%M%S}{path.suffix}"))
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_COLUMNS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def _prediction_row(run_at: str, report: dict, value_bet: str) -> dict:
    probs = report["probabilities"] or {}
    standings = report["standings"] or {}
    weather = report["weather"] or {}
    quote = report["odds"]["selected"] or {}
    xpts = report.get("xpoints") or {}
    opinion = report.get("second_opinion") or {}
    return {
        "run_at": run_at, "league": report["league"], "kickoff": report["kickoff"],
        "home": report["home"], "away": report["away"],
        "p_home": probs.get("Home"), "p_draw": probs.get("Draw"), "p_away": probs.get("Away"),
        **(report["features"] or {}),
        "home_rank": (standings.get("home") or {}).get("rank"), "away_rank": (standings.get("away") or {}).get("rank"),
        "temperature_c": weather.get("temperature_c"), "precipitation_mm": weather.get("precipitation_mm"),
        "wind_speed_ms": weather.get("wind_speed_ms"),
        "home_luck": (xpts.get("home") or {}).get("luck"), "away_luck": (xpts.get("away") or {}).get("luck"),
        "second_opinion_outcome": opinion.get("outcome"), "second_opinion_prob": opinion.get("probability"),
        "bookmaker": quote.get("bookmaker"), "value_bet": value_bet,
        "matches_source": report["sources"]["matches"]["source"],
        "odds_source": report["sources"].get("odds", {}).get("source"),
        "weather_source": (report["sources"].get("weather") or {}).get("source"),
    }


def _signed(value: float) -> str:
    return f"{value:+.1f}".replace("-", "−")


def _enrichment_lines(report: dict, best: dict, attributions: set[str]) -> list[str]:
    """Lignes d'information ajoutées à une alerte : points attendus, deuxième avis, actus blessures."""
    lines = []
    home, away = report["home"], report["away"]
    xpts = report.get("xpoints") or {}
    if xpts.get("home") and xpts.get("away"):
        lines.append(
            f"📈 Points réels − attendus : {telegram.escape(home)} {_signed(xpts['home']['luck'])} · "
            f"{telegram.escape(away)} {_signed(xpts['away']['luck'])}"
        )
        attributions.add(report["sources"]["xpoints"]["attribution"])
    opinion = report.get("second_opinion")
    if opinion:
        verdict = "✅ même issue" if opinion["outcome"] == best["outcome"] else "⚠️ avis différent"
        lines.append(f"🤖 Bet Better : {OUTCOME_FR[opinion['outcome']]} {round(opinion['probability'] * 100)} % ({verdict})")
        attributions.add(report["sources"]["second_opinion"]["attribution"])
    for team in (home, away):
        found = services.news(team)
        for item in (found.data or [])[:2]:
            lines.append(f"📰 {telegram.escape(item['headline'])} ({telegram.escape(item['source'] or '?')}, {item['published_at'][8:10]}/{item['published_at'][5:7]})")
        if found.data:
            attributions.add(found.attribution)
    return lines


def run() -> dict:
    now = datetime.now(timezone.utc)
    run_at = now.isoformat(timespec="seconds")
    horizon = now + timedelta(days=config.DAYS_AHEAD)
    warnings: list[str] = []
    attributions: set[str] = set()

    # 1. Règlement des paris et bilan
    rows = betting.load()
    settled = betting.settle(rows)
    clv_filled = betting.fill_clv(rows)
    betting.save(rows)
    telegram.send(betting.summary(rows))
    already_bet = betting.existing_keys(rows)

    # 2. Recherche des value bets
    alerts: list[str] = []
    analysed = 0
    for code, league in LEAGUES.items():
        matches = services.matches(code)
        odds = services.odds(code)
        for label, res in (("matchs", matches), ("cotes", odds)):
            if res.attribution:
                attributions.add(res.attribution)
            if res.stale or not res.ok:
                reasons = f" Détail : {' ; '.join(res.errors)}." if res.errors else ""
                warnings.append(f"{league.name}, {label} : {res.message}{reasons}")
        if not odds.ok:
            continue
        for event in analysis.group_events(odds.data):
            kickoff = analysis.kickoff_of(event)
            if not (now < kickoff <= horizon):
                continue
            report = analysis.analyze(code, event["home"], event["away"], kickoff, matches, event["quotes"])
            report["sources"]["odds"] = odds.meta()
            analysed += 1
            best = report["value_bets"][0] if report["value_bets"] else None
            _log_prediction(_prediction_row(run_at, report, best["outcome"] if best else ""))
            if not best:
                continue
            match_name = f"{event['home']} vs {event['away']}"
            if betting.bet_key(match_name, event["utc_date"]) in already_bet:
                continue
            quote = report["odds"]["selected"]
            market_view = report["odds"]["market"]
            market_text = (
                f" · marché ({market_view['bookmaker']}, sans marge) : {round(best['market_probability'] * 100)} %"
                if market_view and "market_probability" in best else ""
            )
            alerts.append("\n".join([
                f"⚽️ *{telegram.escape(match_name)}* ({league.name}, {kickoff:%d/%m %H:%M} UTC)",
                f"🎯 {best['outcome']} @ {best['odds']} chez {quote['bookmaker']} "
                f"(IA : {round(best['probability'] * 100)} %{market_text}) | Value : {best['value']}",
                *_enrichment_lines(report, best, attributions),
            ]))
            rows.append({
                "Match": match_name, "Pari": best["outcome"], "Cote": best["odds"], "Value": best["value"],
                "Date": now.strftime("%Y-%m-%d %H:%M"), "Result": "", "Ligue": code,
                "Bookmaker": quote["bookmaker"], "CoupEnvoi": event["utc_date"],
            })
            already_bet.add(betting.bet_key(match_name, event["utc_date"]))
    betting.save(rows)

    # 3. Envoi Telegram (5 alertes par message), avec les mentions de source exigées
    footer = "\n\n_" + " · ".join(sorted(attributions)) + "_" if attributions else ""
    for i in range(0, len(alerts), 5):
        telegram.send("🚀 *NOUVELLES OPPORTUNITÉS IA* 🚀\n\n" + "\n\n".join(alerts[i:i + 5]) + footer)
    if warnings:
        telegram.send("⚠️ *Sources en difficulté*\n" + "\n".join(f"• {w}" for w in dict.fromkeys(warnings)))

    summary = {"settled": settled, "clv_filled": clv_filled, "analysed": analysed, "alerts": len(alerts), "warnings": list(dict.fromkeys(warnings))}
    log.info("Terminé : %s", summary)
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s : %(message)s")
    run()


if __name__ == "__main__":
    main()
