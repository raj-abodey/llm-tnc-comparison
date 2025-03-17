import json

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from utils import read_tnc_files

jan_2015_tnc, mar_2023_tnc = read_tnc_files()

llm_input = jan_2015_tnc + "\n" + mar_2023_tnc
file_names = [
    "./output/baseline_overall.txt",
    "./output/baseline_graphrag.txt",
    "./output/baseline_openaiEmb_semanticChnk.txt",
    "./output/baseline_opensourceEmb_semanticChnk.txt",
    "./output/baseline_ollamaEmb_normalChnk.txt",
    "./output/extractive_summary.txt",
]
result = {}
for file_name in file_names:
    print(f"Processing {file_name}")

    with open(file_name, "r") as file:
        llm_output = file.read()

    correctness_metric = GEval(
        name="Correctness",
        # criteria="Determine whether the actual output is factually correct based on the input.",
        evaluation_steps=[
            "Read the input carefully and identify main points, the differences between the two terms and documents",
            "Read the actual_output and compare it to the input. Check if the output summarises the key differences and assess the significance and impact of the changes, and is presented in a clear and logical order",
            "Assign a score for coherence on a scale of 1 to 10, where 1 is the lowest and 10 is the highest based on the evaluation criteria",
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
    )

    test_case = LLMTestCase(input=llm_input, actual_output=llm_output)
    # To run metric as a standalone
    correctness_metric.measure(test_case)
    print(correctness_metric.score, correctness_metric.reason)
    temp_result = {
        "score": correctness_metric.score,
        "reason": correctness_metric.reason,
    }
    result[file_name] = temp_result
print(result)
json_string = json.dumps(result, indent=4)
with open("./output/geval_result.json", "w") as f:
    print(json_string, file=f)
