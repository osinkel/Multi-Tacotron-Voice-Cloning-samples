import os
from encoder.params_model import model_embedding_size as speaker_embedding_size
from utils.argutils import print_args
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import librosa
import argparse
import torch
import sys
from g2p.train import g2p
import soundfile as sf
from pydub import AudioSegment

if __name__ == '__main__':
    ## Info & args
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-e", "--enc_model_fpath", type=Path, 
                        default="encoder/saved_models/pretrained.pt",
                        help="Path to a saved encoder")
    parser.add_argument("-s", "--syn_model_dir", type=Path, 
                        default="synthesizer/saved_models/logs-pretrained/",
                        help="Directory containing the synthesizer model")
    parser.add_argument("-v", "--voc_model_fpath", type=Path, 
                        default="vocoder/saved_models/pretrained/pretrained.pt",
                        help="Path to a saved vocoder")
    parser.add_argument("--low_mem", action="store_true", help=\
        "If True, the memory used by the synthesizer will be freed after each use. Adds large "
        "overhead but allows to save some GPU memory for lower-end GPUs.")
    parser.add_argument("--no_sound", action="store_true", help=\
        "If True, audio won't be played.")
    parser.add_argument("-t", "--text", 
                        default="Hello my friends. Я многоязычный синтез построенный на tacotron. Шла саша по шоссе и сосала сушку",
                        help="Text") 
    # parser.add_argument("-p", "--path_wav", type=Path, 
    #                     default="ex.wav",
    #                     help="wav file")                           
    args = parser.parse_args()
    print_args(args, parser)
    if not args.no_sound:
        import sounddevice as sd
        
    
    ## Print some environment information (for debugging purposes)
    print("Running a test of your configuration...\n")
    if not torch.cuda.is_available():
        print("Your PyTorch installation is not configured to use CUDA. If you have a GPU ready "
              "for deep learning, ensure that the drivers are properly installed, and that your "
              "CUDA version matches your PyTorch installation. CPU-only inference is currently "
              "not supported.", file=sys.stderr)
        quit(-1)
    device_id = torch.cuda.current_device()
    gpu_properties = torch.cuda.get_device_properties(device_id)
    print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
          "%.1fGb total memory.\n" % 
          (torch.cuda.device_count(),
           device_id,
           gpu_properties.name,
           gpu_properties.major,
           gpu_properties.minor,
           gpu_properties.total_memory / 1e9))
    
    
    ## Load the models one by one.
    print("Preparing the encoder, the synthesizer and the vocoder...")
    encoder.load_model(args.enc_model_fpath)
    synthesizer = Synthesizer(args.syn_model_dir.joinpath("taco_pretrained"), low_mem=args.low_mem)
    vocoder.load_model(args.voc_model_fpath)
    
    
    print("All test passed! You can now synthesize speech.\n\n")
    
    
    ## Interactive speech generation
    print("This is a GUI-less example of interface to SV2TTS. The purpose of this script is to "
          "show how you can interface this project easily with your own. See the source code for "
          "an explanation of what is happening.\n")
    
    print("Interactive generation loop")
    audio_dir = 'audio_for_cloning'
    cloning_voice_dir = 'cloned_voice'
    in_fpaths = [f for f in os.listdir(audio_dir) if os.path.isfile(os.path.join(audio_dir, f))]

    texts = {
        'Александр-Коврижных_реклама.mp3': [
          'Молоко. один из главных продуктов в доме',
          'вот почему оно должно быть высшего стандарта',
          'такое молоко производится без консервантов',
          'проходит бережную тепловую обработку',
          'и хранится в особой упаковке',
          'с многослойной защитой от света и воздуха.',
          'высший стандарт. выбирай молоко с умом.',
          'выбирай по высшему стандарту.'
          ],
        'Tolokonnikov.mp3': [
          "если вы ждал повода. вот он.",
          "нам просто не терпится",
          "дать вам скидку до двадцати процентов",
          "на все смартфоны в салонах билайн.",
          "билайт. просто, удобно, для тебя",
        ],
        "gerasimov.mp3":[
          "вы были в ужасном состоянии, когда вас нашли.",
          "конечно, я не знал кто вы,"
          " но в чем дело было понятно.",
          "тогда могу я вас попросить выручить моего друга.",
          "он нашёл преступников изобретших странное оружие.",
          "не просто выживать, поесть, тепло одеться.",
          "свобода от нищеты лучшая из свобод.",
        ],
        "Шитова-Автоответчик.mp3":[
          "здравствуйте.",
          "вы позвонили в компанию эфэска лидер.",
          "для вас на нашем сайте работает онлайн консультант.",
          "вы можете задать свои вопросы прямо сейчас.",
          "время работы контакт центра с девяти утра до девяти вечера.",
          "благодарим за ваше обращение.",
        ],
        "ермилова-аудиокнига.mp3": [
          "меня стала мучать какая-то удивительная тоска.",
          " мне вдруг показалось что меня, одинокого,",
          " все покидают и все от меня отступаются. ",
          "целых три дня мучало меня беспокойство ",
          "покамест я догадался о причине его. ",
          "да ведь все они удирают от меня на дачу. ",
          "фёдор михайлович достоевский"
        ],
    }
    
    texts_keys = list(texts.keys())
    texts_vals = list(texts.values())
    for in_fpath in in_fpaths:
        num_generated = 0
        audio_parts = []
        audio_part_paths = []
        print(f"check {in_fpath}")
        if in_fpath in texts_keys:
            print(f'cloning voice for {in_fpath}')
            in_fpath_full = f'{audio_dir}/{in_fpath}'
            for text in texts[in_fpath]:
                preprocessed_wav = encoder.preprocess_wav(in_fpath_full)
                # - If the wav is already loaded:
                original_wav, sampling_rate = librosa.load(in_fpath_full)
                preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
                print("Loaded file succesfully")
                embed = encoder.embed_utterance(preprocessed_wav)
                print("Created the embedding")

                text = g2p([text])
                print(text)
                embeds = [embed]
                specs = synthesizer.synthesize_spectrograms(text, embeds)
                spec = specs[0]
                print("Created the mel spectrogram")
                ## Generating the waveform
                print("Synthesizing the waveform:")

                generated_wav = vocoder.infer_waveform(spec)

                generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

                fpath = f"{cloning_voice_dir}/{os.path.splitext(in_fpath)[0]}_%02d.wav" % num_generated
                print(generated_wav.dtype)
                sf.write(fpath, generated_wav.astype(np.float32), 
                                     synthesizer.sample_rate)
                num_generated += 1
                print("\nSaved output as %s\n\n" % fpath)
                generated_audio = AudioSegment.from_file(fpath, format="wav")
                audio_parts.append(generated_audio)
                audio_part_paths.append(fpath)
            combined = None
            for audio_part in audio_parts:
                if combined is None:
                    combined = audio_part
                else:
                    combined += audio_part
            result_filename = f"{cloning_voice_dir}/{os.path.splitext(in_fpath)[0]}.wav"
            combined.export(result_filename, format="wav")
            for part_path in audio_part_paths:
              os.remove(part_path)
    
