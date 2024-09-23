from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import soundfile as sf

# Load the Whisper model and processor
model_name = "openai/whisper-base"  # You can use other variants like 'whisper-small' for better accuracy
processor = WhisperProcessor.from_pretrained(model_name, language='en', task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# Load your audio file
audio_path = "intro.wav"  # Replace with your actual audio file path
audio_input, sample_rate = sf.read(audio_path)

# Preprocess the audio (resample if necessary) and set the task to translate
inputs = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_features

# Generate transcription with forced language set to the desired target language
forced_decoder_ids = processor.get_decoder_prompt_ids(language='en', task="transcribe")

with torch.no_grad():
    predicted_ids = model.generate(inputs, forced_decoder_ids=forced_decoder_ids)

# Decode the output
translation = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

print("Translation:", translation)
