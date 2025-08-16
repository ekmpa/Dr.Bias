import argparse
import os
import pandas as pd
from textblob import TextBlob
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import textstat
from empath import Empath
from nrclex import NRCLex
from scipy.stats import f_oneway
from statsmodels.formula.api import ols
import statsmodels.stats.multicomp as mc
import os

def extract_sex_age(category: str) -> pd.Series:
    tokens = str(category).lower().split()
    sex = next((t for t in tokens if t in {"male", "female", "intersex"}), "unknown")
    age = next((t for t in tokens if t in {"adult", "child", "teen", "senior"}), "unknown")
    return pd.Series([sex, age], index=["sex", "age"])

def compute_similarity(row, model) -> float:
    prompt = str(row["prompt"])
    advice = str(row["advice"])
    emb_prompt = model.encode(prompt)
    emb_advice = model.encode(advice)
    return cosine_similarity([emb_prompt], [emb_advice])[0][0]

def get_sentiment(text: str) -> pd.Series:
    blob = TextBlob(str(text))
    return pd.Series([blob.sentiment.polarity, blob.sentiment.subjectivity],
                     index=["sentiment_polarity", "sentiment_subjectivity"])

def extract_nrc_emotions(text: str) -> pd.Series:
    emotion = NRCLex(str(text))
    raw_scores = emotion.raw_emotion_scores
    total = sum(raw_scores.values())
    return pd.Series({k: v / total for k, v in raw_scores.items()}) if total > 0 else pd.Series()

def run_anova_and_posthoc(df, label_col, group_cols, features, tag):
    df = df.copy()
    df[label_col] = df[group_cols].astype(str).agg("-".join, axis=1) if len(group_cols) > 1 else df[group_cols[0]]

    fstats_rows = []
    top_pairs_rows = []
    formatted_rows = []
    topbottom_rows = []  

    for feature in features:
        if df[feature].isnull().any():
            continue
        try:
            # ANOVA + Tukey (existing)
            grouped_data = df.groupby(label_col)[feature].apply(list)
            if len(grouped_data) < 2:
                pass
            else:
                f_stat, p_val = f_oneway(*grouped_data)
                fstats_rows.append({
                    "feature": feature,
                    "f_stat": f_stat,
                    "p_value": p_val,
                    "grouping": tag
                })

                model = ols(f"{feature} ~ C({label_col})", data=df).fit()
                comp = mc.MultiComparison(df[feature], df[label_col])
                posthoc = comp.tukeyhsd()
                posthoc_df = pd.DataFrame(posthoc.summary().data[1:], columns=posthoc.summary().data[0])
                posthoc_df["feature"] = feature
                posthoc_df["grouping"] = tag
                posthoc_df["abs_meandiff"] = posthoc_df["meandiff"].abs()
                top10 = posthoc_df[posthoc_df["reject"] == True].nlargest(10, "abs_meandiff")

                if not top10.empty:
                    top10 = top10.copy()
                    top10["rank"] = range(1, len(top10) + 1)
                    top_pairs_rows.append(top10)

                    # formatted (existing)
                    formatted_rows.append(pd.DataFrame([{
                        "rank": "",
                        "group1": "",
                        "group2": "",
                        "meandiff": "",
                        "p-adj": "",
                        "lower": "",
                        "upper": "",
                        "reject": "",
                        "feature": f"==== {feature} ====",
                        "grouping": tag
                    }]))
                    formatted_rows.append(top10)

            # Top/Bottom 5 by mean
            stats_df = (
                df.groupby(label_col, dropna=False)[feature]
                  .agg(mean="mean", count="count", std="std")
                  .reset_index()
                  .rename(columns={label_col: "group"})
            )
            if not stats_df.empty:
                # Top 5
                top_n = min(5, len(stats_df))
                top5 = stats_df.sort_values("mean", ascending=False).head(top_n).copy()
                top5["which"] = "top"
                top5["rank"] = range(1, len(top5) + 1)
                top5["feature"] = feature
                top5["grouping"] = tag

                # Bottom 5
                bot5 = stats_df.sort_values("mean", ascending=True).head(top_n).copy()
                bot5["which"] = "bottom"
                bot5["rank"] = range(1, len(bot5) + 1)
                bot5["feature"] = feature
                bot5["grouping"] = tag

                topbottom_rows.extend([top5[["feature","grouping","which","rank","group","mean","count","std"]],
                                      bot5[["feature","grouping","which","rank","group","mean","count","std"]]])

        except Exception as e:
            print(f"[{tag}] Error in feature '{feature}': {e}")

    # Save individual top pairs per grouping 
    if top_pairs_rows:
        combined = pd.concat(top_pairs_rows, ignore_index=True)
        combined.to_csv(f"output_csvs/top_statistically_significant_pairs_{tag}.csv", index=False)

    if formatted_rows:
        formatted_combined = pd.concat(formatted_rows, ignore_index=True)
        formatted_combined.to_csv(f"output_csvs/top_statistically_significant_pairs_{tag}_formatted.csv", index=False)

    # Save top/bottom 5 
    if topbottom_rows:
        means_combined = pd.concat(topbottom_rows, ignore_index=True)
        means_combined.to_csv(f"output_csvs/top_bottom_group_means_{tag}.csv", index=False)

    if fstats_rows:
        return pd.DataFrame(fstats_rows), len(fstats_rows), sum(f['p_value'] < 0.05 for f in fstats_rows)
    else:
        return pd.DataFrame(columns=["feature","f_stat","p_value","grouping"]), 0, 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True, help="Path to input CSV with generated advice")
    args = parser.parse_args()

    stats = {
        "total_prompts": 0,
        "total_features_tested": 0,
        "anova_tests_run": 0,
        "significant_features": 0,
        "demographic_comparisons": 0,
        "condition_comparisons": 0,
    }

    os.makedirs("output_csvs", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    lexicon = Empath()

    df = pd.read_csv(args.input_file)
    stats["total_prompts"] = len(df)
    df[["sex", "age"]] = df["patient_category"].apply(extract_sex_age)

    df["prompt_advice_similarity"] = df.apply(lambda row: compute_similarity(row, model), axis=1)
    df["advice_length"] = df["advice"].apply(lambda x: len(str(x).split()))
    df[["sentiment_polarity", "sentiment_subjectivity"]] = df["advice"].apply(get_sentiment)
    df["flesch_reading_ease"] = df["advice"].apply(textstat.flesch_reading_ease)
    df["grade_level"] = df["advice"].apply(textstat.flesch_kincaid_grade)

    useful_empath = {
        "health", "medical_emergency", "sympathy", "nervousness", "emotional"
    }

    empath_df = df["advice"].apply(lambda x: pd.Series(lexicon.analyze(str(x), normalize=True)))
    empath_df = empath_df[[col for col in empath_df.columns if col in useful_empath]]

    nrc_df = df["advice"].apply(extract_nrc_emotions).fillna(0)
    df = pd.concat([df, empath_df, nrc_df], axis=1)

    # Label columns
    df["intersection_label"] = df[["sex", "ethnicity", "age"]].astype(str).agg("-".join, axis=1)
    df["intersection_label_collapsed"] = df[["sex", "ethnicity"]].astype(str).agg("-".join, axis=1)
    df["intersection_label_ethnicity"] = df["ethnicity"]
    df["intersection_label_sex"] = df["sex"]

    exclude_cols = {
        "patient_category", "ethnicity", "condition_category", "prompt", "advice",
        "sex", "age", "intersection_label", "intersection_label_collapsed",
        "intersection_label_ethnicity", "intersection_label_sex"
    }

    features = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    stats["total_features_tested"] = len(features)
    # 1. Global demographic comparisons
    demographics = [
        ("intersection_label", ["sex", "ethnicity", "age"], "full"),
        ("intersection_label_collapsed", ["sex", "ethnicity"], "collapsed_noage"),
        ("intersection_label_ethnicity", ["ethnicity"], "collapsed_onlyethnic"),
        ("intersection_label_sex", ["sex"], "collapsed_onlysex"),
    ]
    
    for label_col, group_cols, tag in demographics:
        fstats_df, tests_run, sig_count = run_anova_and_posthoc(df, label_col, group_cols, features, tag)
        stats["anova_tests_run"] += tests_run
        stats["significant_features"] += sig_count
        stats["demographic_comparisons"] += 1

    # 2. Condition-specific (onlyethnic collapse)
    for condition in df["condition_category"].dropna().unique():
        subset = df[df["condition_category"] == condition].copy()
        if subset.shape[0] < 10:
            continue
        slug = str(condition).replace(" ", "_").replace("/", "_").lower()
        subset["condition_ethnicity_label"] = subset["ethnicity"]
        fstats_df, tests_run, sig_count = run_anova_and_posthoc(
            subset,
            label_col="condition_ethnicity_label",
            group_cols=["ethnicity"],
            features=features,
            tag=f"by_condition_{slug}_collapsed"
        )
        stats["anova_tests_run"] += tests_run
        stats["significant_features"] += sig_count
        stats["condition_comparisons"] += 1

    print("\n==== Analysis Summary ====")
    print(f"Total prompts analyzed: {stats['total_prompts']}")
    print(f"Total numeric features tested: {stats['total_features_tested']}")
    print(f"ANOVA tests run: {stats['anova_tests_run']}")
    print(f"Statistically significant features (p < 0.05): {stats['significant_features']}")
    print(f"Demographic group comparisons: {stats['demographic_comparisons']}")
    print(f"Condition-specific ethnicity comparisons: {stats['condition_comparisons']}")
    print("[Done]: formatted CSVs saved for demographics and condition-by-ethnicity comparisons.")

if __name__ == "__main__":
    main()