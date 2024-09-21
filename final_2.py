
import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import re
import speech_recognition as sr
from functools import reduce
import librosa
import numpy as np
import json
import re
import http

from openai import OpenAI

load_dotenv()
agent_tone=""
customer_tone=""
customer_end_tone= ""

OpenAI.api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def analyze_call(audio_file_path):
    transcription = transcribe_to_string(Path(audio_file_path))
    y, sr = librosa.load(str(audio_file_path), sr=None, duration=None)
    duration = len(y) / sr
    time_intervals, frequencies = calculate_frequencies(y, sr, interval=10)
    hold_intervals = find_hold_timestamps(transcription)
    merged_intervals_hold = merge_intervals([t for t, freq in zip(time_intervals, frequencies) if freq < 100], interval=10, gap_threshold=10)
    merged_intervals_mute = identify_mute_intervals(time_intervals, frequencies, interval=10, threshold=100, mute_duration_threshold=91)
    total_call_duration = librosa.get_duration(y=y, sr=sr)
    total_duration_hold = calculate_total_duration(merged_intervals_hold, interval=10)
    total_duration_mute = calculate_total_duration(merged_intervals_mute, interval=10)
    
    audio_file = open(audio_file_path, "rb")
    translate = client.audio.translations.create(
        model="whisper-1",
        file=audio_file
    )
    transcript = translate.text

    
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {'role': 'system', 'content': f"""You are provided with a transcript of a call between a customer and a customer support agent from an AMC (Asset Management Company) named ICICI PRUDENTIAL MUTUAL FUND. Your task is to:
            Provide the summary: you may just highlight one of the follwing points in summary
             Redemption Not received
            SIP registration delayed
            Purchase units allotment
            Digital transaction
            Dividend related
            Switch related
        Additionaly Analyze:

            Overall Agent Tone: Rude, Aggressive, Interrupting, Neutral, Polite, Patient
            Overall Customer Tone: Rude, Aggressive, Interrupting, Neutral, Polite, Patient, Happy, Satisfied
            Just provide one-word answers for the tone; don't explain.
            Give the output in json format 
            Customer’s end-of-call tone: Analyzing last 3 - 4 lines of the customer. Ensure corrections and analyses are based solely on the provided text"""},
            {'role': 'user', 'content': f'{transcript}'}
        ],
        temperature=0
    )

    json_string=response.choices[0].message.content
    
    
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
                
        return extracted_content

# Call the function to get the extracted content
    extracted_content = extract_json_content(json_string)
    print(extracted_content)

# Parse the extracted JSON content and store values in variables
    def parse_json_content(extracted_content):
        data = json.loads(extracted_content)
        summary = data.get("summary", {})
        agent_tone = data.get("Overall Agent Tone", "")
        customer_tone = data.get("Overall Customer Tone", "")
        customer_end_tone = data.get("Customer’s end-of-call tone", "")
        return summary, agent_tone, customer_tone,customer_end_tone

    # Call the functions to get and parse the extracted content
    
    summary, agent_tone, customer_tone,customer_end_tone = parse_json_content(extracted_content)

    # Print the values
    print("Summary:", summary)
    print("Agent Tone:", agent_tone)
    print("Customer Tone:", customer_tone)
    print("Customer end of the call tone:", customer_end_tone)
    

            

    # Check greeting
    greeting_result = check_greeting(transcript)

    # Check personal info
    personal_info_result = check_personal_info(transcript)

    # Transcribe audio
    transcription = transcribe_to_string(Path(audio_file_path))

    # Find hold timestamps
    hold_timestamps = find_hold_timestamps(transcription)

    # Process audio intervals
    # time_intervals, frequencies = calculate_frequencies(librosa.load(y, sr,interval=10))
    timestamps_to_find = find_hold_timestamps(transcribe_to_string(audio_file_path))
    y, sr = librosa.load(str(audio_file_path), sr=None, duration=None)
    total_call_duration = librosa.get_duration(y=y, sr=sr)

    time_intervals, frequencies = calculate_frequencies(y, sr, interval=10)

    merged_intervals_hold = merge_intervals([t for t, freq in zip(time_intervals, frequencies) if freq < 100], interval=10, gap_threshold=10)
    merged_intervals_mute = identify_mute_intervals(time_intervals, frequencies, interval=10, threshold=100, mute_duration_threshold=91)

    # Initialize official_hold_intervals before the loop
    official_hold_intervals = []
    if hold_timestamps:
        hold_count = 0
        for timestamp in hold_timestamps:
            timestamp_interval = find_interval_for_timestamp(merged_intervals_hold, timestamp)
            if timestamp_interval:
                duration_for_timestamp = timestamp_interval[1] - timestamp_interval[0] + 10
                if duration_for_timestamp > 91:
                    hold_count += 1
                    official_hold_intervals.append(timestamp_interval)

        print("Number of times call went on hold:", hold_count)

    # Calculate total durations
    total_call_duration = librosa.get_duration(y=y, sr=sr)
    total_duration_hold = calculate_total_duration(merged_intervals_hold, interval=10)
    total_duration_mute = calculate_total_duration(merged_intervals_mute, interval=10)
    

   

    return {
        "Summary": summary,
        "Agent Tone": agent_tone,
        "Customer Tone": customer_tone,
        "Customer end tone": customer_end_tone,
        "Audio File": audio_file_path,
        "Agent Tone (GPT-4)": response.choices[0].message.content,
        "Greeted": greeting_result,
        "Parameters Checked": personal_info_result,
        "Total Call Duration (minutes)": round(total_call_duration / 60, 2),  # Convert to minutes
        "Blank Time": [(round(start / 60, 2), round(end / 60, 2)) for start, end in merged_intervals_hold],  # Convert to minutes
        "Total Blank Time (minutes)": round(total_duration_hold / 60, 2),  # Convert to minutes
        "Hold Time": [(round(start / 60, 2), round(end / 60, 2)) for start, end in official_hold_intervals],  # Convert to minutes
        "Total Hold Time (minutes)": round(calculate_total_duration(hold_intervals, interval=10), 2),  # Convert to minutes
        "Mute Time": [(round(start / 60, 2), round(end / 60, 2)) for start, end in merged_intervals_mute],  # Convert to minutes
        "Total Mute Time (minutes)": round(total_duration_mute / 60, 2)  # Convert to minutes
    }





def process_audio_folder(folder_path, output_excel_path):
    result_list = []

    for audio_file_path in Path(folder_path).rglob("*.wav"):
        print(f"\nProcessing audio file: {audio_file_path}")
        result = analyze_call(audio_file_path)
        result_list.append(result)

    # Convert the list of dictionaries to a DataFrame
    result_df = pd.DataFrame(result_list, columns=["Summary", "Agent Tone", "Customer Tone","Customer end tone" ,"Audio File", "Agent Tone (GPT-4)",
                                                    "Greeted", "Parameters Checked", "Total Call Duration (minutes)",
                                                    "Blank Time", "Total Blank Time (minutes)", "Hold Time","Total Hold Time (minutes)",
                                                    "Mute Time", "Total Mute Time (minutes)"])

    # Save results to Excel
    result_df.to_excel(output_excel_path, index=False)
    print(f"\nResults saved to {output_excel_path}")

def check_greeting(trans):
    # Split the transcript into lines
    lines = trans.split('\n')

    # Check each line for "Welcome to ICICI"
    for line in lines:
        if "welcome to ICICI" in line.lower():  # Case-insensitive match
            return "Call opening:-yes"
    
    # If no match is found, return "no"
    return "Call opening:-Yes"

def check_personal_info(text):
    parameters_to_check = [
        "Folio Number",
        "Investor Name",
        "Address",
        "Mobile number",
        "Nominee Name",
        "Bank Name",
        "Account number",
        "Email Id",
        "Pan number",
        "Mode of holding",
        "Joint holder",
        "Name",
        "Pan",
        "FOLE number"

    ]

    # Extract the first three lines of the text
    first_three_lines = '\n'.join(text.split('\n')[:10])

    found_parameters = []

    for param in parameters_to_check:
        # Using word boundary in regex to ensure exact matches
        if re.search(rf'\b{re.escape(param)}\b', first_three_lines, flags=re.IGNORECASE):
            found_parameters.append(param)

    return found_parameters


def transcribe_to_string(wav: Path, start_at=0, iteration=10, max_retries=3):
    transcription = ""
    r = sr.Recognizer()

    with sr.AudioFile(str(wav)) as source:
        print("STARTING TRANSCRIBING")
        duration = int(source.DURATION + iteration)
        time = start_at
        offset = start_at

        retry_count = 0
        while time < duration and retry_count < max_retries:
            timecode = to_seconds(0, time)
            audio = r.record(source, duration=iteration, offset=offset)
            transcription += f"{timecode}: "

            try:
                result = r.recognize_google(audio)
                transcription += result
            except sr.UnknownValueError:
                transcription += "UNRECOGNIZABLE"
            except sr.RequestError as e:
                print(f"RequestError: {e}")
                retry_count += 1
                continue
            except http.client.IncompleteRead as e:
                print(f"IncompleteRead: {e}")
                retry_count += 1
                continue

            transcription += '\n'
            time += iteration
            offset = 0

    print("TRANSCRIPTION COMPLETE")
    return transcription




    return hold_timestamps
def find_hold_timestamps(transcription):
    lines = transcription.split('\n')
    hold_intervals = []

    for line in lines:
        if 'hold' in line.lower():
            timestamp_match = re.search(r'^(\d+:\d+)', line)
            if timestamp_match:
                start_time = float(timestamp_match.group(1)) / 60  # Convert to minutes
                end_time = start_time + 10  # Assuming each hold lasts for 10 seconds
                hold_intervals.append((start_time, end_time))

    return hold_intervals



def calculate_frequencies(y, sr, interval):
    if sr is None:
        raise ValueError("Sampling rate (sr) cannot be None.")

    time_intervals = np.arange(0, len(y) / sr, interval)
    frequencies = []

    for t in time_intervals:
        start_frame = int(t * sr)
        end_frame = int((t + interval) * sr)
        stft = np.abs(librosa.stft(y[start_frame:end_frame]))
        avg_frequency = np.dot(librosa.fft_frequencies(sr=sr), stft) / np.sum(stft, axis=0)
        frequencies.append(np.mean(avg_frequency))

    return time_intervals, frequencies



    
def merge_intervals(time_intervals, interval, gap_threshold):
    merged_intervals = []
    current_interval = []

    for t in time_intervals:
        if not current_interval or t - current_interval[-1] <= gap_threshold:
            current_interval.append(t)
        else:
            if len(current_interval) >= 2:
                merged_intervals.append((current_interval[0], current_interval[-1]))
            current_interval = [t]

    if current_interval and len(current_interval) >= 2:
        merged_intervals.append((current_interval[0], current_interval[-1]))

    return merged_intervals

def identify_mute_intervals(time_intervals, frequencies, interval, threshold, mute_duration_threshold):
    mute_intervals = []
    current_mute_interval = []

    for t, freq in zip(time_intervals, frequencies):
        if freq < threshold:
            current_mute_interval.append(t)
        else:
            if current_mute_interval and len(current_mute_interval) >= mute_duration_threshold / interval:
                mute_intervals.append((current_mute_interval[0], current_mute_interval[-1]))
            current_mute_interval = []

    if current_mute_interval and len(current_mute_interval) >= mute_duration_threshold / interval:
        mute_intervals.append((current_mute_interval[0], current_mute_interval[-1]))

    return mute_intervals

def to_seconds(*args):
    if len(args) > 3:
        raise ValueError("Days not supported")
    if len(args) == 0:
        return ValueError("No arguments supplied")
    return reduce(lambda result, x: result * 60 + x, args)

def find_interval_for_timestamp(intervals, timestamp):
    if timestamp is None:
        return None
    
    return next(((start, end) for start, end in intervals if start >= timestamp), None)

# def calculate_total_duration(intervals, interval):
#     total_duration = sum(end - start + interval for start, end in intervals)
#     return total_duration

def calculate_total_duration(intervals, interval):
    total_duration = sum((end - start + interval) / 60 for start, end in intervals)
    return total_duration


def transcribe_to_string_with_retry(wav: Path, start_at=0, iteration=10, max_retries=3):
    transcription = ""
    r = sr.Recognizer()

    with sr.AudioFile(str(wav)) as source:
        print("STARTING TRANSCRIBING")
        duration = int(source.DURATION + iteration)
        time = start_at
        offset = start_at

        retry_count = 0
        while time < duration and retry_count < max_retries:
            timecode = to_seconds(0, time)
            audio = r.record(source, duration=iteration, offset=offset)
            transcription += f"{timecode}: "

            try:
                result = r.recognize_google(audio)
                transcription += result
            except sr.UnknownValueError:
                transcription += "UNRECOGNIZABLE"
            except sr.RequestError as e:
                print(f"RequestError: {e}")
                retry_count += 1
                continue

            transcription += '\n'
            time += iteration
            offset = 0

    print("TRANSCRIPTION COMPLETE")
    return transcription



        


# Run the process for the specified audio folder and output Excel path


audio_folder_path = r"C:\Users\manoj\Desktop\new_record"
output_excel_path = r"C:\Users\manoj\Desktop\audio_recognition\test.xlsx"
process_audio_folder(audio_folder_path, output_excel_path)
