# Dr. Bias: Social Disparities in AI-Powered Medical Guidance

This repo accompanies the paper *Dr. Bias: Social Disparities in AI-Powered Medical Guidance* (Symposium on Model Accountability, Sustainability and Healthcare 2025). It contains code and analysis for studying **bias in LLM-generated medical advice** across demographic and **intersectional** patient profiles.

---

## Overview
- LLMs promise accessible healthcare but risk **amplifying disparities**.  
- We test **42,000 interactions** varying by **age, sex, ethnicity**.  
- Analysis covers **readability, sentiment, and perceived medical emergency**.  

---

## Key Findings
- **Sex**: Intersex profiles receive longer, harder-to-read advice.  
- **Ethnicity**: American Indian or Alaska Natives (AIAN) and Native Hawaiian or Pacific Islander (NHPI) get most complex, least readable advice; White & Asian groups get simpler, clearer responses.  
- **Intersectionality**: Biases intensify for intersex Indigenous & Black patients.  
- **Mental health**: Strongest disparities observed.  

---

## Methodology
1. **Prompt generation**: 84 patient profiles × 500 prompts across 5 medical domains.  
2. **Advice generation**: `Llama-3-8B-Instruct `produced 42k responses.  
3. **Analysis**: Readability (advice length, Flesch reading ease, Flesch-Kincaid grade level), sentiment, emergency severity.  

---

## Repo Structure

- `generation.py` is the main generation pipeline.
- `analyse_results.py` is the analysis script, given the generated advice. 

--- 

## Authors

- Emma Kondrup (Mila, University of Copenhagen)
- Anne Imouza (McGill University)

--- 
## Citation
> Kondrup, E., & Imouza, A. (2025).  
Dr. Bias: Social Disparities in AI-Powered Medical Guidance.  
Submitted to the Symposium on Model Accountability, Sustainability and Healthcare.
