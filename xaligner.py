import whisperx
import gc 
from tqdm import tqdm
from fuzzysearch import find_near_matches
from datetime import datetime, timedelta
from pathlib import Path
import re
from num2words import num2words
from unidecode import unidecode
from argparse import ArgumentParser

parser = ArgumentParser()  
parser.add_argument("--duration", '-d', type=float, default=100)
parser.add_argument("--audio_file", '-a')
parser.add_argument("--text_file", '-t')
args = parser.parse_args()

def digit_to_string(match):
    digits = match.group()
    return str(num2words(digits)).replace('-',' ')
def seconds_to_hh_mm_ss_ms(seconds):
    # Calculate total seconds and milliseconds
    total_seconds = int(seconds)
    milliseconds = int((seconds - total_seconds) * 1000)

    # Create a timedelta object representing the duration
    duration = timedelta(seconds=total_seconds, milliseconds=milliseconds)

    # Use the timedelta to create a datetime object
    dt = datetime(1, 1, 1) + duration

    # Format the datetime object as hh:mm:ss,ms
    formatted_time = dt.strftime('%H:%M:%S,%f')[:-3]

    return formatted_time

device = "cuda" 
audio_file = "Mustache.wav"
batch_size = 4 # reduce if low on GPU mem
compute_type = "int8" # change to "int8" if low on GPU mem (may reduce accuracy)

model = whisperx.load_model("medium", device, compute_type=compute_type, language='en')
model_a, metadata = whisperx.load_align_model(language_code='en', device=device)
def main():
    audio_file = Path(args.audio_file)
    text_file = Path(args.text_file)
    srt_file = audio_file.parent/f'{audio_file.stem}.srt'
    with open(text_file, encoding='utf-8') as f:
        canon = f.read()
    # canon = unidecode(canon)
    audio = whisperx.load_audio(str(audio_file))
    # audio = audio[:16_000*100]
    duration = args.duration
    chunk_size = 16_000*duration
    def data(audio):
        for i in range(0, audio.shape[0], chunk_size):
            yield audio[i:i+chunk_size]

    offset=0
    max_l_dist=0
    word_segments_total=[]
    for chunk in tqdm(data(audio), total=len(range(0, audio.shape[0], chunk_size))):
        result = model.transcribe(chunk, batch_size=batch_size, language='en')
        print(result["segments"])
        # start_pos
        first_seg = result["segments"][0]['text']
        print(f'finding the start\n{first_seg}')
        start_match=[]
        while len(start_match)==0:
            start_match = find_near_matches(first_seg.lower(), canon.lower(), max_l_dist=max_l_dist)
            max_l_dist+=1
            if len(start_match)==0 and max_l_dist==1: print(f'increasing max_l_dist')
        print('start found')
            
        # reset max_l_dist
        max_l_dist = 0
        
        # end_pos
        last_seg = result["segments"][-1]['text']
        print(f'finding the end\n{last_seg}')
        end_match=[]
        while len(end_match)==0:
            end_match = find_near_matches(last_seg.lower(), canon.lower(), max_l_dist=max_l_dist)
            max_l_dist+=1
            if len(end_match)==0 and max_l_dist==1: print(f'increasing max_l_dist')
        print('end found')

        # reset max_l_dist
        max_l_dist = 0
        
        # use this to align
        trans = (canon[start_match[0].start: end_match[0].end])
        trans = re.sub(r'\n+', ' ', trans)
        # replace numerals
        # trans = re.sub(r'\d+', digit_to_string, trans)
        transcript = [{
            'text':trans,
            'start':0,
            'end':duration
        }]
        # print(f'doing this part:\n{trans}')
        resultAlign = whisperx.align(transcript, model_a, metadata, chunk, device, return_char_alignments=False)
        # print([w['word'] for w in resultAlign['word_segments']])
        offset_align = []
        for dict in resultAlign['word_segments']:
            try:
                offset_align.append({
                    'word': dict['word'], 'start':dict['start']+offset, 'end':dict['end']+offset
                })
            except:
                print(f'alignment not found for {dict["word"]}. Skipping')

            # offset_align = [{'word': dict['word'], 'start':dict['start']+offset, 'end':dict['end']+offset} for dict in resultAlign['word_segments']]
        word_segments_total.extend(offset_align)
        offset+=duration
        print(f'truncating canon from {len(canon)}')
        canon = canon[end_match[0].end:]
        print(f'to {len(canon)} characters')

    srt=''
    index=1
    for word in word_segments_total:
        start = seconds_to_hh_mm_ss_ms(word['start'])
        end = seconds_to_hh_mm_ss_ms(word['end'])
        w = word['word'].replace('\n', ' ')
        # print('*'*20)
        # print(w)
        # print('*'*20)
        srt+=f'{index}\n'
        srt+=f"{start} --> {end}\n"
        srt+=f'{w}\n\n'
        index+=1
    with open(srt_file, 'w', encoding='utf-8') as f:
        f.write(srt)
if __name__=='__main__':
    main()