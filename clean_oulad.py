"""
clean_oulad.py

יוצר את data/tma_submissions.csv מתוך קבצי OULAD המקוריים שב-data/oulad/.
כל שורה: סטודנט אחד ועבודה כתובה (TMA) אחת שהוא היה צריך להגיש, ומה קרה איתה.
ההחלטות מאחורי כל שלב מופיעות ב-docs/decisions.md, סעיפים 5 ו-7.

הרצה מתיקיית הפרויקט:  python clean_oulad.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "oulad"
OUT = ROOT / "data" / "tma_submissions.csv"

COURSE_KEYS = ["code_module", "code_presentation"]   # קורס + סמסטר
REG_KEYS = COURSE_KEYS + ["id_student"]               # רישום אחד של סטודנט לקורס

# התחום של כל קורס, לפי המאמר של OULAD
DOMAIN = {
    "AAA": "Social Sciences", "BBB": "Social Sciences", "GGG": "Social Sciences",
    "CCC": "STEM", "DDD": "STEM", "EEE": "STEM", "FFF": "STEM",
}

# העמודות בקובץ הנקי, לפי הסדר. final_result והציון לא נכנסים (דליפת מידע).
COLUMNS = [
    # מזהים ופרטי הקורס
    "id_student", "code_module", "code_presentation", "id_assessment",
    "domain", "year", "start_month",
    # פרטי העבודה והעומס (קלטים למודל)
    "tma_number", "n_tmas", "deadline_day", "weight", "studied_credits",
    # נתונים דמוגרפיים (לדשבורד בלבד)
    "gender", "age_band", "highest_education", "region", "imd_band", "disability",
    # מה קרה בהגשה
    "date_submitted", "days_from_deadline", "status", "is_late",
    # היסטוריה מעבודות קודמות באותו קורס (קלטים למודל)
    "prev_status", "prev_late_rate",
]


def read(name):
    # ב-OULAD ערכים חסרים כתובים כ-"?"
    return pd.read_csv(RAW / f"{name}.csv", na_values="?")


def main():
    assessments = read("assessments")
    info = read("studentInfo")
    registration = read("studentRegistration")
    submissions = read("studentAssessment")

    # 1. רק עבודות כתובות (TMA), עם מספר לפי סדר הדדליינים בקורס
    tma = assessments[assessments["assessment_type"] == "TMA"].copy()
    tma = tma.sort_values(COURSE_KEYS + ["date"])
    tma["tma_number"] = tma.groupby(COURSE_KEYS).cumcount() + 1
    tma["n_tmas"] = tma.groupby(COURSE_KEYS)["id_assessment"].transform("size")
    tma = tma.rename(columns={"date": "deadline_day"})
    tma["deadline_day"] = tma["deadline_day"].astype(int)   # לכל העבודות יש דדליין
    tma = tma[COURSE_KEYS + ["id_assessment", "tma_number", "n_tmas", "deadline_day", "weight"]]

    # 2. "מי היה צריך להגיש מה": כל רישום לקורס, כפול כל עבודה בקורס
    students = info.merge(registration, on=REG_KEYS, how="inner")
    rows = students.merge(tma, on=COURSE_KEYS, how="inner")
    n_all = len(rows)

    # משאירים רק מי שהיה רשום ביום הדדליין.
    # תאריך חסר נחשב כ"לא ידוע", ולכן השורה נשארת.
    registered_after = rows["date_registration"] > rows["deadline_day"]
    left_by_deadline = rows["date_unregistration"] <= rows["deadline_day"]
    rows = rows[~registered_after & ~left_by_deadline]

    # 3. מצמידים את ההגשות. ציון שהועבר מניסיון קודם (is_banked) הוא לא הגשה בסמסטר הזה.
    rows = rows.merge(
        submissions[["id_assessment", "id_student", "date_submitted", "is_banked"]],
        on=["id_assessment", "id_student"],
        how="left",
    )
    banked = rows["is_banked"] == 1
    rows = rows[~banked].copy()

    # 4. מה קרה בהגשה. איחור = הוגשה אחרי יום הדדליין (הגשה ביום הדדליין עצמו היא בזמן).
    rows["date_submitted"] = rows["date_submitted"].astype("Int64")
    rows["days_from_deadline"] = rows["date_submitted"] - rows["deadline_day"]
    submitted = rows["date_submitted"].notna()
    late = (rows["days_from_deadline"] > 0).fillna(False).astype(bool)
    rows["status"] = np.select([~submitted, late], ["not_submitted", "late"], default="on_time")
    rows["is_late"] = late.astype("Int64").where(submitted)   # ריק אם לא הוגשה

    # 5. היסטוריה: רק מעבודות קודמות של אותו סטודנט באותו קורס,
    #    כך שהמודל רואה רק מה שהיה ידוע לפני הדדליין הנוכחי.
    rows = rows.sort_values(REG_KEYS + ["tma_number"]).reset_index(drop=True)
    groups = rows.groupby(REG_KEYS, sort=False)
    rows["prev_status"] = groups["status"].shift().fillna("first")

    rows["_late"] = (rows["status"] == "late").astype(int)
    rows["_submitted"] = (rows["status"] != "not_submitted").astype(int)
    groups = rows.groupby(REG_KEYS, sort=False)
    prev_late = groups["_late"].cumsum() - rows["_late"]
    prev_submitted = groups["_submitted"].cumsum() - rows["_submitted"]
    rows["prev_late_rate"] = (prev_late / prev_submitted).where(prev_submitted > 0).round(3)

    # 6. פרטים לסינון בדשבורד
    rows["domain"] = rows["code_module"].map(DOMAIN)
    rows["year"] = rows["code_presentation"].str[:4].astype(int)
    rows["start_month"] = rows["code_presentation"].str[4].map({"B": "February", "J": "October"})

    # 7. שמירה
    clean = rows[COLUMNS]
    clean.to_csv(OUT, index=False)

    # 8. בדיקה: המספרים צריכים להיות קרובים למה שכתוב ב-docs/decisions.md
    print(f"Rows before filters:            {n_all:,}")
    print(f"  removed, registered after deadline: {registered_after.sum():,}")
    print(f"  removed, left on/before deadline:   {left_by_deadline.sum():,}")
    print(f"  removed, banked from earlier try:   {banked.sum():,}")
    print(f"Final rows:                     {len(clean):,}")
    print(f"Students: {clean['id_student'].nunique():,}   Assignments (TMA): {clean['id_assessment'].nunique()}")
    print("Status shares (%):", (clean["status"].value_counts(normalize=True) * 100).round(1).to_dict())
    print(f"Late among submitted: {clean['is_late'].mean() * 100:.1f}%")
    print(f"Saved: {OUT.relative_to(ROOT)}  ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
