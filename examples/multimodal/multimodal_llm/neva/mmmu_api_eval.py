from datasets import load_dataset, get_dataset_config_names
import ast, os, json, tqdm, base64, requests, re

api_key = "dummy"
api_base = "http://10.110.40.142:8000/chat/completions"
dataset_name = "MMMU/MMMU"
name = "mmmu_v1_all"
answers = {}

for config in tqdm.tqdm(get_dataset_config_names(dataset_name)):
 for split in ["test"]:
  dataset = load_dataset("MMMU/MMMU", config, split=split)
  # Iterate over the validation split
  for e in dataset:
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

    #if e["image_2"] is None:
    #   continue

    #print(f"ID: {mmmu_id}\nPrompt: {prompt}, Expected Response: {answer}, {e}\n\n")

    # save image to disk and update the prompt
    for i in range(1, 4):
        if e[f"image_{i}"] is not None:
            filename = os.path.join(name, f"{mmmu_id}_image_{i}.jpg")
            url = f"https://pdx.s8k.io/v1/AUTH_nevada/{filename}"
            if not os.path.exists(filename):
                e[f"image_{i}"].convert("RGB").save(filename)


    #prompt = prompt.replace("<image 1>", f"\n{img}\n").strip()

    # Call OpenAI GPT4v API
    api_key = "dummy"
    headers = {"Content-Type": "application/json","Authorization": f"Bearer {api_key}"}
    payload = {
        "model": "neva",
        "messages": [{"role": "user","content": []}],
        "max_tokens": 500
    }

    # Match the <image [0-9]> tags and the text in the prompt with a regex, and split it into parts
    # to add {"type": "text", "text": a.strip()} and {"type": "image_url", "image_url": {"url": url}} in between
    parts = re.split(r'<image (\d+)>', prompt)
    content = payload["messages"][0]["content"]

    # We are using this flag to add the image_url only once for now. TODO...
    flag = False
    
    for i in range(len(parts)):
        if i % 2 == 0:                          # Even indices are text parts        
            text_part = parts[i].strip()
            if text_part:
                content.append({"type": "text", "text": text_part})
        else:
            filename = os.path.join(name, f"{mmmu_id}_image_{parts[i]}.jpg")
            url = f"https://pdx.s8k.io/v1/AUTH_nevada/{filename}"

            # Add the image_url only once for now. TODO.
            if not flag:            
                content.append({"type": "image_url", "image_url": {"url": url}})
                flag = True

    #print(payload)
    # Call the API via requests/POST
    response = requests.post(api_base, headers=headers, json=payload)
    #print(response.json())

    # Extract the model response
    model_response = response.json()["choices"][0]["message"]["content"]
   

    #{
    #"validation_Accounting_1": "D", # strictly "A", "B", "C", "D" for multi-choice question
    #"validation_Architecture_and_Engineering_14": "0.0", # any string response for open question.
    #}
    last = model_response.strip().rsplit("\n", 1)[-1]
    answer = None
    if options:
        for letter in last.split():
            letter = letter.strip(".").strip(":")
            if letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                 answer = letter
                 break
        if answer is None:
            guessing = [letter for letter in "ABCD" if letter in last]
            guessing += [letter for letter in "ABCD" if letter in model_response]
            guessing += ["A"]
            answer = guessing[0]

    else:
        answer = last.strip(".").strip()
            

    answers[mmmu_id] = answer
    #print(f"ID: {mmmu_id}, URL: {url}\nPrompt: {prompt}, Expected Response: {answer}, Model Response: {model_response}\n\n") 

    print(answer, "\n", model_response, "\n\n")
    with open(f"{name}-mmmu-answers-preview10-nolabel.json", "w") as f:
        json.dump(answers, f)

