import gradio as gr
from transformers import pipeline


classifier = pipeline(
    "text-classification",
    model="Shunian/mbti-classification-bert-base-uncased",
    top_k=1,
)


def analytics_emo(x):
    data = classifier(x)[0]
    label = data[0]["label"]
    iamge = "images/{}.png".format(label)
    return iamge


# input is a text box
# output is an image
gr.Interface(
    fn=analytics_emo,
    inputs=gr.Textbox(
        lines=5, placeholder="Type your text here...", label="Text input"
    ),
    outputs=gr.Image(
        type="pil",
        label="MBTI Personality",
    ),
    title="MBTI Personality Test",
    description="This is a demo of MBTI Personality Test",
    allow_flagging=False,
    # examples for the user to try
    examples=[
        "My only recent picture of me is thanks to my ninja colleague Raged She still refuses to delete it"
    ],
).launch()
