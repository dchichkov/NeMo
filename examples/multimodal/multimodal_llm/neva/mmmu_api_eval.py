# This script evaluates the MMMU (val) using the OpenAI-compatible API
# To launch the vLLM server, use the following command:
# CONTAINER=vllm/vllm-openai:latest
# 
# mkdir -p ~/pixtral/cache/huggingface
# docker run -it \
#         --gpus 0,1,2,4 \
#         --shm-size=16g \
#         -v ~/pixtral/cache/huggingface:/root/.cache/huggingface \
#         -p 8000:8000 \
#         --ipc=host \
#         --ulimit memlock=-1 \
#         --ulimit stack=67108864 \
#         --env "HUGGING_FACE_HUB_TOKEN=hf_YOUR_TOKEN" \
#         $CONTAINER --model mistralai/Pixtral-12B-2409 --tokenizer_mode mistral --limit_mm_per_prompt 'image=4' --tensor-parallel-size 2 --max-model-len 8192
#

from datasets import load_dataset, get_dataset_config_names
import ast, os, json, tqdm, base64, requests, re, string, ray, backoff, collections

dataset_name = "MMMU/MMMU"
name = "mmmu_v1_all"
if os.path.exists(f"{name}-answers.json"):
    questions = json.load(open(f"{name}-answers.json"))
elif os.path.exists(f"{name}.questions.json"):
    questions = json.load(open(f"{name}.questions.json"))
else:
    counter = collections.Counter()
    questions = collections.defaultdict(list)
    for config in tqdm.tqdm(get_dataset_config_names(dataset_name)):
        for split in ["validation"]:
            for e in load_dataset(dataset_name, config, split=split):
                options = ast.literal_eval(e["options"])
                letters = string.ascii_uppercase[:len(options)]
                loptions = [f"{letter}. {option}" for option, letter in zip(options, letters)]
                questions[config].append({"question": e["question"], "options": options, "loptions": loptions, "letters": letters, "expected": e["answer"], "id": e["id"]})
                for i in range(1, 9):
                    if e[f"image_{i}"] is not None:
                       filename = os.path.join(name, f"{e['id']}_image_{i}.jpg")
                       url = f"https://pdx.s8k.io/v1/AUTH_nevada/{filename}"
                       if not os.path.exists(filename):
                           e[f"image_{i}"].convert("RGB").save(filename)
                    else: break
                counter[i] += 1
    print("Images per question:", counter.most_common(10))

    with open(f"{name}.questions.json", "w") as f:    
        json.dump(questions, f)



@ray.remote
def process_item(e):
    mmmu_id, options, loptions, expected = e["id"], e["options"], e["loptions"], e["expected"]

    if options:
        #prompt = e["question"] + "\n\nOptions:\n" +'\n'.join(loptions) + "\n\nSolve it (step by step when nessesary). Finally, output the letter of the correct option inside double curly."
        #prompt = e["question"] + "\n\nOptions:\n" +'\n'.join(loptions) + "\n\nAnswer with the letter of the correct option."
        #prompt = e["question"] + "\n\nOptions:\n" +'\n'.join(loptions) + "\n\nSolve it (step by step when necessary). Finally, output the letter of the correct option inside double curly braces on a new line. For example, if the correct option is 'A', output {{A}}. Follow this format exactly."
        prompt = e["question"] + "\n\nOptions:\n" +'\n'.join(loptions) + "\n\nSolve it (step by step when necessary). Finally, output the letter of the correct option inside double curly braces on a new line. For example, if the correct option is 'A', output {{A}}. Follow this format exactly on a single line, do not use LaTeX in the final answer."
    else:
        #prompt = e["question"] + "\n\nSolve it (step by step when nessesary). Finally, output the answer only with no explanations."
        prompt = e["question"] + "\n\nOutput reasoning steps and give the final answer inside double curly braces on a new line. For example, if the answer is '42.1', output {{42.1}}.  Follow this format exactly on a single line, do not use LaTeX in the final answer."

    # Call OpenAI GPT4v API
    api_key = "dummy"
    api_base = "http://10.110.40.142:8000/v1/chat/completions"
    headers = {"Content-Type": "application/json","Authorization": f"Bearer {api_key}"}
    payload = {
        "model": "mistralai/Pixtral-12B-2409",
        "messages": [{"role": "user","content": []}],
        "max_tokens": 2048
    }

    # Match the <image [0-9]> tags and the text in the prompt with a regex, and split it into parts
    # to add {"type": "text", "text": a.strip()} and {"type": "image_url", "image_url": {"url": url}} in between
    parts = re.split(r'<image (\d+)>', prompt)
    content = payload["messages"][0]["content"]
    

    for i in range(len(parts)):
        if i % 2 == 0:                          # Even indices are text parts        
            text_part = parts[i].strip()
            if text_part:
                content.append({"type": "text", "text": text_part})
        else:
            filename = os.path.join(name, f"{mmmu_id}_image_{parts[i]}.jpg")
            url = f"https://pdx.s8k.io/v1/AUTH_nevada/{filename}"

            # Add the image_url, up to 4 images
            # if len([c for c in content if c["type"] == "image_url"]) < 4:
            content.append({"type": "image_url", "image_url": {"url": url}})
    
    # Do the API call
    @backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_time=60)
    def call_api():
        return requests.post(api_base, headers=headers, json=payload, timeout=60)
    print(f"Response for {mmmu_id}: {call_api().json()}")
    model_response = call_api().json()["choices"][0]["message"]["content"]
    return mmmu_id, model_response

def parse_result(e):
    mmmu_id, options, letters, expected, model_response = e["id"], e["options"], e["letters"], e["expected"], e["model_response"]

    # Extract the answer from the model response
    result = model_response.strip().split("\n\n")[-1].strip()
    result = result.replace("\}", "}").replace("\{", "{")
    result = result.replace("```", "").replace("\n", "")
    if "{{" in result and "}}" in result: result = result.split("{{", 1)[1].split("}}", 1)[0]
    elif "{" in result and "}" in result: result = result.split("{", 1)[1].split("}", 1)[0]

    verbose = False
    if not result:
        result = 'A'
        if verbose: print(f"Defaulting to 'A' when no answer is available, for: {model_response}")

    if not options and ("{{" not in model_response or "}}" not in model_response):
        if verbose: print("The format of the answer was not followed for:", model_response)

    if options:  # "letter of the correct option at the end, in double curly braces." in j["prompt"]:
        if result not in string.ascii_uppercase:
            result = ([l for l in result if l in letters] + ["A"])[0]
            #print(f"The answer should be a single letter. Using guessed answer for: {model_response}\nResult:", result)          
            
    if len(result.split()) > 10:
        result = result.split()[-1].strip(".").strip()
        if verbose: print(f"The answer should be short. Using last word for: {model_response}\nResult:", result)

    return result


# up to 8 workers
ray.init(num_cpus=12)

correct, total = 0, 0

# concat into single list from questions.values()
sample = sum(questions.values(), [])[::10]

#for config_questions in [tqdm.tqdm(sum(questions.values(), [])[::10])]:
for config_questions in tqdm.tqdm(questions.values()):
    futures = [process_item.remote(e) for e in config_questions if "model_response" not in e]
    model_responses = dict(ray.get(futures))  

    for e in config_questions:
        if "model_response" not in e:
            e["model_response"] = model_responses[e["id"]]

        e["result"] = parse_result(e)
        mmmu_id, expected, result, model_response = e["id"], e["expected"], e["result"], e["model_response"]

        if e["result"] == e["expected"]:
            correct += 1
        else:
            print(f"Model Response: {model_response}\nID: {mmmu_id}, Result: {result}, Expected: {expected}\n\n")
        total += 1
        # print(result)
    print(f"Correct: {correct}/{total} ({correct/total:.3%})")  

    with open(f"{name}-answers.json", "w") as f:
        json.dump(questions, f, indent=4)
  

