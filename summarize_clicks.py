"""
summarize_clicks.py

מסכם את קובץ הקליקים הגדול (data/oulad/studentVle.csv, 433MB) לקובץ קטן:
data/clicks_by_day.csv
בכל שורה: קורס, עבודה, יום ביחס לדדליין, וקבוצה (הגישו בזמן / באיחור / לא הגישו),
וכמה סטודנטים מהקבוצה נכנסו לאתר הקורס באותו יום.
בשביל הגרף "מתי עובדים" ובשביל הכלל של "מתי להתחיל".
ההחלטות: docs/decisions.md, סעיף 7.

צריך להריץ קודם את clean_oulad.py.
הרצה מתיקיית הפרויקט:  python summarize_clicks.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CLICKS = ROOT / "data" / "oulad" / "studentVle.csv"
SUBMISSIONS = ROOT / "data" / "tma_submissions.csv"
OUT = ROOT / "data" / "clicks_by_day.csv"

REG_KEYS = ["code_module", "code_presentation", "id_student"]
WINDOW_BEFORE = 30   # כמה ימים לפני הדדליין
WINDOW_AFTER = 14    # כמה ימים אחרי הדדליין
CHUNK = 2_000_000


def daily_clicks():
    """סך הקליקים לכל סטודנט, בכל קורס, בכל יום. קורא את הקובץ הגדול בחלקים."""
    parts = []
    for chunk in pd.read_csv(CLICKS, usecols=REG_KEYS + ["date", "sum_click"], chunksize=CHUNK):
        parts.append(chunk.groupby(REG_KEYS + ["date"], as_index=False)["sum_click"].sum())
    # סטודנט-יום יכול להתפצל בין שני חלקים, ולכן מסכמים שוב
    return pd.concat(parts).groupby(REG_KEYS + ["date"], as_index=False)["sum_click"].sum()


def main():
    subs = pd.read_csv(SUBMISSIONS, usecols=REG_KEYS + [
        "id_assessment", "tma_number", "domain", "deadline_day", "status"])
    daily = daily_clicks()
    print(f"Student-days with clicks: {len(daily):,}")

    # לכל עבודה: הימים בחלון סביב הדדליין. עובדים עבודה-עבודה (לפי המספר בקורס) כדי לחסוך זיכרון.
    pieces = []
    for n in sorted(subs["tma_number"].unique()):
        rows = subs[subs["tma_number"] == n]
        days = rows.merge(daily, on=REG_KEYS, how="inner")
        days["days_from_deadline"] = days["date"] - days["deadline_day"]
        days = days[days["days_from_deadline"].between(-WINDOW_BEFORE, WINDOW_AFTER)]
        pieces.append(
            days.groupby(["id_assessment", "status", "days_from_deadline"], as_index=False)
                .agg(active_students=("id_student", "nunique"), total_clicks=("sum_click", "sum"))
        )
    activity = pd.concat(pieces)

    # טבלה מלאה: כל עבודה x כל קבוצה x כל יום בחלון, גם ימים שאף אחד לא נכנס בהם (0)
    groups = (subs.groupby(["code_module", "code_presentation", "domain", "id_assessment",
                            "tma_number", "status"], as_index=False)
                  .size().rename(columns={"size": "students_in_group"}))
    all_days = pd.DataFrame({"days_from_deadline": range(-WINDOW_BEFORE, WINDOW_AFTER + 1)})
    grid = groups.merge(all_days, how="cross")
    out = grid.merge(activity, on=["id_assessment", "status", "days_from_deadline"], how="left")
    out[["active_students", "total_clicks"]] = out[["active_students", "total_clicks"]].fillna(0).astype(int)
    out["pct_active"] = (out["active_students"] / out["students_in_group"] * 100).round(1)
    out = out.rename(columns={"status": "group"})
    out = out[["code_module", "code_presentation", "domain", "id_assessment", "tma_number", "group",
               "days_from_deadline", "students_in_group", "active_students", "pct_active", "total_clicks"]]
    out = out.sort_values(["code_module", "code_presentation", "tma_number", "group", "days_from_deadline"])
    out.to_csv(OUT, index=False)

    # בדיקה: איזה חלק מהקליקים של 30 הימים שלפני הדדליין נעשה ב-3 הימים האחרונים (2- עד 0).
    # אם העבודה הייתה מתפזרת שווה, זה היה 3 מתוך 31 ימים, כלומר כ-10%.
    before = out[out["days_from_deadline"].between(-WINDOW_BEFORE, 0)]
    last3 = before["days_from_deadline"] >= -2
    print(f"Rows: {len(out):,}   Assignments: {out['id_assessment'].nunique()}")
    print(f"Share of pre-deadline clicks in last 3 days: {before.loc[last3, 'total_clicks'].sum() / before['total_clicks'].sum() * 100:.1f}%  (even spread: {3 / 31 * 100:.1f}%)")
    by_group = before.groupby("group").apply(
        lambda g: round(g.loc[g["days_from_deadline"] >= -2, "total_clicks"].sum() / g["total_clicks"].sum() * 100, 1))
    print("  by group (%):", by_group.to_dict())
    print(f"Saved: {OUT.relative_to(ROOT)}  ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
