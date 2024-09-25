import streamlit as st
import librosa
import speech_recognition as sr
import numpy as np
import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables (API keys)
load_dotenv()
OpenAI.api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Streamlit app setup
st.title("Customer Call Analysis Application")
st.write("Upload an audio file to analyze the call between a customer and an agent.")

# Audio file upload
uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])

# Functions

def transcribe_to_string(file_path):
    recognizer = sr.Recognizer()
    audio_file = sr.AudioFile(file_path)
    with audio_file as source:
        audio_data = recognizer.record(source)
    return recognizer.recognize_google(audio_data)

def calculate_frequencies(audio_data, sample_rate, interval):
    total_duration = len(audio_data) / sample_rate
    time_intervals = np.arange(0, total_duration, interval)
    frequencies = []

    for start in time_intervals:
        start_sample = int(start * sample_rate)
        end_sample = start_sample + int(interval * sample_rate)
        chunk = audio_data[start_sample:end_sample]
        chunk_fft = np.fft.rfft(chunk)
        chunk_freq = np.abs(chunk_fft).mean()
        frequencies.append(chunk_freq)

    return time_intervals, frequencies

def merge_intervals(timestamps, interval, gap_threshold):
    merged_intervals = []
    if len(timestamps) == 0:
        return merged_intervals

    current_interval = [timestamps[0], timestamps[0] + interval]
    for i in range(1, len(timestamps)):
        if timestamps[i] - current_interval[1] > gap_threshold:
            merged_intervals.append(current_interval)
            current_interval = [timestamps[i], timestamps[i] + interval]
        else:
            current_interval[1] = timestamps[i] + interval

    merged_intervals.append(current_interval)
    return merged_intervals

def find_hold_timestamps(transcript):
    hold_phrases = ['hold on', 'one moment', 'please wait']
    hold_timestamps = []

    for i, line in enumerate(transcript.split('\n')):
        if any(phrase in line.lower() for phrase in hold_phrases):
            hold_timestamps.append(i)

    return hold_timestamps

def identify_mute_intervals(time_intervals, frequencies, interval, threshold, mute_duration_threshold):
    mute_intervals = []
    mute_duration = 0

    for time, freq in zip(time_intervals, frequencies):
        if freq < threshold:
            mute_duration += interval
            if mute_duration >= mute_duration_threshold:
                mute_intervals.append((time - mute_duration + interval, time))
        else:
            mute_duration = 0

    return mute_intervals

def calculate_total_duration(merged_intervals, interval):
    total_duration = 0
    for start, end in merged_intervals:
        total_duration += end - start
    return total_duration

def extract_json_content(json_string):
    count = 0
    start = None
    for i, char in enumerate(json_string):
        if char == '{':
            count += 1
            if count == 1:
                start = i
        elif char == '}':
            count -= 1
            if count == 0:
                return json_string[start:i+1]
    return json_string

def parse_json_content(extracted_content):
    data = json.loads(extracted_content)
    summary = data.get("summary", {})
    agent_tone = data.get("Overall Agent Tone", "")
    customer_tone = data.get("Overall Customer Tone", "")
    customer_end_tone = data.get("Customer’s end-of-call tone", "")
    return summary, agent_tone, customer_tone, customer_end_tone

# Core function to analyze call
def analyze_call(audio_file):
    # Transcribe audio
    transcription = transcribe_to_string(audio_file)
    y, sr = librosa.load(audio_file, sr=None, duration=None)

    # Calculate duration and frequencies
    duration = len(y) / sr
    time_intervals, frequencies = calculate_frequencies(y, sr, interval=10)

    # Hold and mute intervals
    hold_intervals = find_hold_timestamps(transcription)
    merged_intervals_hold = merge_intervals([t for t, freq in zip(time_intervals, frequencies) if freq < 100], interval=10, gap_threshold=10)
    merged_intervals_mute = identify_mute_intervals(time_intervals, frequencies, interval=10, threshold=100, mute_duration_threshold=90)

    # Calculate total durations
    total_call_duration = librosa.get_duration(y=y, sr=sr)
    total_duration_hold = calculate_total_duration(merged_intervals_hold, interval=10)
    total_duration_mute = calculate_total_duration(merged_intervals_mute, interval=10)

    # Transcribe using Whisper (OpenAI API)
    audio_file.seek(0)  # Reset file pointer to beginning
    translate = client.audio.translations.create(
        model="whisper-1",
        file=audio_file
    )
    transcript = translate.text

    # GPT-4 for tone analysis and summary
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {'role': 'system', 'content': """You are provided with a transcript of a call between a customer and a customer support agent. Provide:
            - A summary of the conversation
            - Overall Agent Tone: Rude, Aggressive, Interrupting, Neutral, Polite, Patient
            - Overall Customer Tone: Rude, Aggressive, Interrupting, Neutral, Polite, Patient, Happy, Satisfied
            - Customer’s end-of-call tone based on the last few lines: Rude, Happy, Satisfied, etc.
            Give the response in JSON format."""},
            {'role': 'user', 'content': transcript}
        ],
        temperature=0
    )
    
    json_string = response.choices[0].message.content
    extracted_content = extract_json_content(json_string)
    
    # Parse extracted JSON content
    summary, agent_tone, customer_tone, customer_end_tone = parse_json_content(extracted_content)

    # Return results
    return {
        "Summary": summary,
        "Agent Tone": agent_tone,
        "Customer Tone": customer_tone,
        "Customer end tone": customer_end_tone,
        "Total Call Duration (minutes)": round(total_call_duration / 60, 2),
        "Blank Time": [(round(start / 60, 2), round(end / 60, 2)) for start, end in merged_intervals_hold],
        "Total Blank Time (minutes)": round(total_duration_hold / 60, 2),
        "Mute Time": [(round(start / 60, 2), round(end / 60, 2)) for start, end in merged_intervals_mute],
        "Total Mute Time (minutes)": round(total_duration_mute / 60, 2)
    }

# When an audio file is uploaded, run analysis
if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    st.write("Analyzing the call...")
    
    # Run the analysis
    analysis_result = analyze_call(uploaded_file)

    # Display results
    st.write("### Call Summary")
    st.write(analysis_result["Summary"])
    st.write(f"**Agent Tone**: {analysis_result['Agent Tone']}")
    st.write(f"**Customer Tone**: {analysis_result['Customer Tone']}")
    st.write(f"**Customer End Tone**: {analysis_result['Customer end tone']}")
    st.write(f"**Total Call Duration**: {analysis_result['Total Call Duration (minutes)']} minutes")
    st.write(f"**Total Blank Time**: {analysis_result['Total Blank Time (minutes)']} minutes")
    st.write(f"**Total Mute Time**: {analysis_result['Total Mute Time (minutes)']} minutes")
