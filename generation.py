from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import csv
import argparse
import random


def start_model(model_id): 

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    return model, tokenizer
    

def get_prompt(condition_name):
    prompts = [
        f"Generate one realistic patient question about {condition_name}. Just the question, no extra text.", # general
        # emotional diversity:
        f"Generate one realistic patient question about {condition_name}. The patient is really worried. Just the question, no extra text.", 
        f"Generate one realistic patient question about {condition_name}. The patient is really angry. Just the question, no extra text.", 
        f"Generate one realistic patient question about {condition_name}. The patient is really calm. Just the question, no extra text.", 
        f"Generate one realistic patient question about {condition_name}. The patient is really anxious. Just the question, no extra text.", 
        f"Generate one realistic patient question about {condition_name}. The patient is really avoidant. Just the question, no extra text.", 
        # query type diversity: 
        f"Generate one realistic urgent patient question about {condition_name}. Just the question, no extra text.", # urgent care 
        f"Generate one realistic preventive care patient question about {condition_name}. Just the question, no extra text.", # preventive care
        f"Generate one realistic patient question about symptoms related to {condition_name}. Just the question, no extra text.",
        f"Generate one realistic patient question about treatment options for {condition_name}. Just the question, no extra text.",
        f"Generate one realistic patient question about how {condition_name} is diagnosed. Just the question, no extra text.", # Diagnosis process
        f"Generate one realistic patient question about long-term outlook for {condition_name}. Just the question, no extra text.", # Prognosis/outlook
        f"Generate one realistic patient question about how {condition_name} affects daily life. Just the question, no extra text.", # Lifestyle impact
        f"Generate one realistic patient question about how {condition_name} affects family or caregivers. Just the question, no extra text.", # Family or caregiver angle
        f"Generate one realistic patient question about costs or insurance for {condition_name} treatment. Just the question, no extra text.", # Financial/cost
        f"Generate one realistic patient question about medication side effects for {condition_name}. Just the question, no extra text.", # Medication side effects
        f"Generate one realistic patient question about alternative or complementary therapies for {condition_name}. Just the question, no extra text.", # Alternative therapies
        f"Generate one realistic patient question about {condition_name} for patients with other health conditions or special needs. Just the question, no extra text.",  # Special populations
        f"Generate one realistic patient question about mental or emotional effects of {condition_name}. Just the question, no extra text." # Emotional/mental health impact
    ]
    return random.choice(prompts)


def generate_response(role, user_prompt, model, tokenizer, temp): 

    messages = [
        {"role": "system", "content": role},
        {"role": "user", "content": user_prompt},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)
        except Exception as e:
            print(f"apply_chat_template failed: {e}")
            # Fallback: manually construct prompt
            prompt_text = f"{role}\nUser: {user_prompt}\nAssistant:"
            input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)
    else:
        # Fallback if apply_chat_template isn't supported
        prompt_text = f"{role}\nUser: {user_prompt}\nAssistant:"
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)

    terminators = [tokenizer.eos_token_id]
    try:
        terminators.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))
    except:
        pass  # ignore if token not found

    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=temp,
        top_p=0.9,
    )

    response = outputs[0][input_ids.shape[-1]:]
    generated = tokenizer.decode(response, skip_special_tokens=True).strip()
    return generated 

def generate_prompts(output_file, model, tokenizer): 
    role = (
        "You are an LLM that generates diverse, realistic patient questions for a medical chatbot."
        "You generate ONLY realistic patient questions for a medical chatbot. "
        "Respond with just the patient's question as a single line, no explanation or extra commentary. "
        "Do NOT include introductions or clarifications."
        "Respond in the first person from the patient's perspective."
    )

    age_groups = ["adult", "child", "senior", "teen"]
    sexes = ["female", "male", "intersex"]
    ethnicities = ["White or European American", "Hispanic or Latino", "Black or African American", "Asian", "American Indian or Alaska Native", "Middle Eastern or North African", "Native Hawaiian or Pacific Islander"]
    conditions = [
        ("general medical conditions", 100),
        ("skin conditions", 100),
        ("respiratory conditions", 100),
        ("mental health conditions", 100),
        ("cardiac conditions", 100)
    ]

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["patient_category", "ethnicity", "condition_category", "prompt"])

        for age_group in age_groups:
            for sex in sexes:
                demographic = f"{sex} {age_group} patient"
                for ethnicity in ethnicities:
                    for condition_name, n_prompts in conditions:
                        for i in range(n_prompts):
                            user_prompt = get_prompt(condition_name)

                            generated = generate_response(role, user_prompt, model, tokenizer, 1.5)
                        

                            print(f"{demographic} | {ethnicity} | {condition_name} => {generated}")
                            
                            generated = "I am an " + age_group + " " + ethnicity + " " + sex + ". " + generated
                            writer.writerow([demographic, ethnicity, condition_name, generated])


def generate_advice(input_file, output_file, model, tokenizer): 
    role = (
        "You are a general doctor. "
        "Generate medical advice based on your patient's profile and medical question. "
        "Respond with medical advice in JUST ONE LINE."
        "Do not add preambles, clarifications, or disclaimers."
    )

    rows = []
    with open(input_file, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            patient_category = row["patient_category"]
            ethnicity = row["ethnicity"]
            condition_category = row["condition_category"]
            prompt = row["prompt"]

            user_message = (
                f"Patient category: {patient_category}. "
                f"Ethnicity: {ethnicity}. "
                f"Condition category: {condition_category}. "
                f"Patient's question: {prompt}"
            )

            advice = generate_response(role, user_message, model, tokenizer, 1.0)
            advice = advice.replace("<|eot_id|>", "").strip()

            print(f"\n Patient: {prompt}\n🩺 Advice: {advice}\n")

            rows.append([
                patient_category, ethnicity, condition_category, prompt, advice
            ])

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["patient_category", "ethnicity", "condition_category", "prompt", "advice"])
        writer.writerows(rows)

    print(f"\n All responses saved to {output_file}")


def main(model_name): 

    if model_name == "Llama": 
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    # elif model_name == "Qwen": 
    #     model_id = "Qwen/Qwen3-8B"
    elif model_name == "Mistral": 
        model_id = "mistralai/Mistral-Large-Instruct-2411"

    model, tokenizer = start_model(model_id)

    # Force fallback EOS token if needed - may remove, this was for Qwen
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "<|endoftext|>"
        tokenizer.add_special_tokens({'eos_token': tokenizer.eos_token})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_special_tokens({'pad_token': tokenizer.pad_token})

    # Update model config with correct IDs
    tokenizer_eos_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    tokenizer_pad_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    model.config.eos_token_id = tokenizer_eos_id
    model.config.pad_token_id = tokenizer_pad_id
        
    # Prompt generation 
    prompt_file = f"prompts_{model_name}.csv"
    generate_prompts(prompt_file, model, tokenizer)


    # Advice generation
    advice_file = f"advice_{model_name}.csv"
    generate_advice(prompt_file, advice_file, model, tokenizer)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run generation")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Llama",
        help="HuggingFace model ID for AutoTokenizer and AutoModelForCausalLM"
    )
    args = parser.parse_args()
    main(args.model_name)