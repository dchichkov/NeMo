import json, string

# TODO: add re-evaluation for longer samples

lines = open("340b_preview10delsys_nv_dpo_mmmu.jsonl", 'r')
results = {}
for line in lines:
    j = json.loads(line)
    answer = j["text"]
    if "Answer with the letter of the correct option" in j["prompt"]:
        result = answer.split()[0].strip(".")
        #letters = 
        if result not in string.ascii_uppercase:
            result = None
            for X in string.ascii_uppercase:
                if f"{X}." in answer or f"{X}:" in answer:
                    result = X
            if not result:
                print(f"Using setting default 'A' answer for: {answer}")
                result = "A"
                #print(answer)
    else:
        result = answer.split("\n")[0].strip()
        if result.startswith("."):
            result = '0' + result 

        if not result and answer.strip():
            result = answer.strip()

        if not result:
            print(f"Defaulting to 0 when no answer is available")
            result = '0'

        if result.endswith("."):
            result = result.rstrip(".")
            
        if result.endswith("0,"):
            print(f"Defaulting to ,000 when answer was cut with ,")
            result += "000"
            print("Result:", result, "Original:", answer)

        if result.endswith(",0"):
            print(f"Defaulting to ,000 when answer was cut with ,0")
            result += "00"
            print("Result:", result, "Original:", answer)

        if result.endswith(",00"):
            print(f"Defaulting to ,000 when answer was cut with ,00")
            result += "0"
            print("Result:", result, "Original:", answer)


         #   print("ANSWER:", letter)

    results[j['question_id']] = result

print(f"Final results file contains {len(results)} results.")

with open("340b_preview10delsys_nv_dpo_mmmu.json", 'w') as f:
    json.dump(results, f)

