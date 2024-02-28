!pip install openai
!pip install pyyaml
!pip install datasets
!pip install matplotlib

# RUN TO ACTIVATE THE VIRTUAL ENV --  source logienv/bin/activate
from openai import OpenAI
import yaml
import matplotlib.pyplot as plt
from datasets import load_dataset
import time
import json

# #READING THE API TOKEN
# with open("config.yaml") as f:
#     config_yaml = yaml.load(f, Loader=yaml.FullLoader)


#CONSTRUCUTING THE OPENAI API
client = OpenAI(
    api_key="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",
)

#READ THE DATASET
dataset = load_dataset("lucasmccabe/logiqa", split = 'train')
rows = dataset[:3]


stop = 0
nonStop = 0
correctAnswers = 0
wrongAnswers = 0

for i in range(len(rows['context'])):
    message = [
        {
            "role": "user",
            "content":f"""
                CONTEXT:{rows['context'][i]}
                QUERY:{rows['query'][i]}
                OPTIONS:{rows['options'][i]}.
                Can you provide me the right answer from the options, your answer should be one of the index in options array.
                Answer should be only in JSON format, for example, answer_index: 1, reason: the reason to choose 1 as answer"""
              },
        ]
    try:
        answer = client.chat.completions.create(
            model = "gpt-3.5-turbo",
            max_tokens = 2048,
            messages = message
        )

        print('\n\n--------------------------------------------------')
        print(rows['context'][i])
        print('\n\n--------------------------------------------------')
        print("GPT-ANSWER",answer.choices[0].message.content)
        print("DATASET-ANSWER",rows['correct_option'][i])
        gpt_answer = json.loads(answer.choices[0].message.content)
        gpt_answer_index = gpt_answer["answer_index"]
        gpt_answer_reason = gpt_answer["reason"]

        if answer.choices[0].finish_reason != "stop":
            stop = stop + 1
        else:
            nonStop = nonStop + 1

        if gpt_answer_index == int(rows['correct_option'][i]):
            correctAnswers = correctAnswers + 1
        else:
          try:
            second_message = [
                  {
            "role": "user",
            "content":f"""
                CONTEXT:{rows['context'][i]}
                QUERY:{rows['query'][i]}
                OPTIONS:{rows['options'][i]}
                REASON:{gpt_answer_reason}
                TYPE_OF_FAILURE:{"transcription error", "miscommunications", "error in stepwise calculations", "parsing error"}
                Previously I asked this query along with the context and options, but you provided {gpt_answer_index} as answer with the REASON mentioned above.
                But it is incorrect, now classify the REASON into one of the TYPE_OF_FAILURE. Give the TYPE_OF_FAILURE in JSON format, for example TYPE_OF_FAILURE: miscommunications, reason: why is it miscommunication type of failure, give me the reason.
            """
              },
            ]
            second_answer = client.chat.completions.create(
              model = "gpt-3.5-turbo",
              max_tokens = 2048,
              messages = second_message
            )
            print("SECOND-GPT-ANSWER",second_answer.choices[0].message.content)
            wrongAnswers = wrongAnswers + 1

          except Exception as e:
            print(f"Rate limit exceeded for second answer {e}")
            time.sleep(30)
        print("correctAnswers",correctAnswers)
        print("wrongAnswers",wrongAnswers)

    except Exception as e:
        print(f"Rate limit exceeded {e}")
        time.sleep(30)

print("correctAnswers",correctAnswers)
print("wrongAnswers",wrongAnswers)

# Data
categories = ['Correct Answers', 'Wrong Answers']
values = [correctAnswers, wrongAnswers]

# Create bar graph.
plt.bar(categories, values)

# Customizations
plt.title('ChatGPT-3.5-Turbo Model Answers on Logical Reasoning Questions')
plt.xlabel('Answers')
plt.ylabel('Number of Answers')

# Display the plot
plt.show()
