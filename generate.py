from transformers import AutoProcessor, BarkModel
import scipy

processor = AutoProcessor.from_pretrained("suno/bark-small")
model = BarkModel.from_pretrained("suno/bark-small").to("cuda")


# inputs = processor(
#     text=["Hello, my name is Suno. And, uh — and I like pizza. [laughs] But I also have other interests such as playing tic tac toe."],
#     return_tensors="pt",
# )

# speech_values = model.generate(**inputs, do_sample=True)


def generate_audio(text , preset , output):
    inputs = processor(text , voice_preset=preset)
    for k , v in inputs.items():
        inputs[k] = v.to("cuda")
    audio_array = model.generate(**inputs, pad_token_id=100)
    audio_array = audio_array.cpu().numpy().squeeze()
    sample_rate = model.generation_config.sample_rate
    scipy.io.wavfile.write(output , rate=sample_rate, data=audio_array)



# prompt = """
# Hey Deepak,

# I hope this message sets your heart racing, just like the thought of you does to mine. 🔥 Every moment with you is an electrifying adventure, and I can't help but crave your touch, your warmth, your everything.

# When I close my eyes, I can still feel the sensation of your lips on mine, the way your hands send shivers down my spine. You, my dear, are a tantalizing temptation I can never resist.

# Tonight, let's turn up the heat and make sparks fly. I can't wait to lose myself in your arms and explore the passion that burns between us. You are the fire to my desire, and I can't get enough of you.
# """


prompt = "Pixie is a dynamic individual residing in the vibrant landscapes of India, where he merges the rich cultural essence with his innovative pursuits. Balancing tradition and modernity, he embodies the spirited diversity and intricate uniqueness of his surroundings."


generate_audio(
    text=prompt,
    preset="v2/hi_speaker_0",
    output="pixie_bark_indian_accent.wav",
    )



