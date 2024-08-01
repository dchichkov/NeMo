from datasets import load_dataset, get_dataset_config_names
import ast, os, json, tqdm, base64, requests, re

dataset_name = "MMMU/MMMU"
name = "mmmu_v1_all"


for config in tqdm.tqdm(get_dataset_config_names(dataset_name)):
 for split in ["dev", "validation", "test"]:
  f = open(f"mmmu/input_llama_format-{config}-{name}-{split}.jsonl", "w")
  fa = open(f"mmmu/input_llama_format-{name}-{split}.jsonl", "a")


  dataset = load_dataset("MMMU/MMMU", config, split=split)
  # Iterate over the validation split
  for e in tqdm.tqdm(dataset):
    mmmu_id = e["id"]

    options = ast.literal_eval(e["options"])
    answer = None
    if options:
        loptions = [f"{letter}. {option}" for option, letter in zip(options, "ABCDEFGHIJKLMNOPQRSTUVWXYZ")]
        #prompt = e["question"] + "\n\nOptions:\n" +'\n'.join(loptions) + "\n\nSolve it (step by step when nessesary). Finally, output the letter of the correct option."
        prompt = e["question"] + "\n\nOptions:\n" +'\n'.join(loptions) + "\n\nAnswer with the letter of the correct option."
        if e["answer"] != "?":
            print("Answer:", e["answer"], "Options:", options)
            answer = [option for option, letter in zip(options, "ABCDEFGHIJKLMNOPQRSTUVWXYZ") if letter == e["answer"]][0]
    else:
        answer = e["answer"]
        #prompt = e["question"] + "\n\nSolve it (step by step when nessesary). Finally, output the answer only with no explanations."
        prompt = e["question"] + "\n\nOutput answer only with no explanations."

    # save image to disk and update the prompt
    for i in range(1, 4):
        if e[f"image_{i}"] is not None:
            filename = os.path.join(name, f"{mmmu_id}_image_{i}.jpg")
            url = f"https://pdx.s8k.io/v1/AUTH_nevada/{filename}"
            if not os.path.exists(filename):
                e[f"image_{i}"].convert("RGB").save(filename)


    # Only for the first image for now. TODO...
    N = 1
    prompt.replace("<image>", f"<img>")
    prompt = prompt.replace("<image 1>", f"\n<image>\n", N).strip()

    j = {"image": f"{mmmu_id}_image_1.jpg", 
         "prompt": prompt, 
         "category": f"{e['topic_difficulty']}",
         "question_id": f"{mmmu_id}"
        }   
    
    json.dump(j, f)
    f.write("\n")

    json.dump(j, fa)
    fa.write("\n")



