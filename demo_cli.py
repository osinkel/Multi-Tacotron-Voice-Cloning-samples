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
    num_generated = 0
    audio_dir = 'audio_for_cloning'

    in_fpaths = [f for f in os.listdir(audio_dir) if os.path.isfile(os.path.join(audio_dir, f))]

    texts = {
        'Александр-Коврижных_реклама.mp3': 'Молоко. один из главных продуктов в доме.\nвот почему оно должно быть высшего стандарта.\nтакое молоко производится без консервантов \nпроходит бережную тепловую обработку \nи хранится в особой упаковке с многослойной защитой от света и воздуха. высший стандарт. выбирай молоко с умом. выбирай по высшему стандарту.'
    }

    texts_keys = list(texts.keys())
    texts_vals = list(texts.values())
    for in_fpath in in_fpaths:
        if in_fpath in texts_keys:
            preprocessed_wav = encoder.preprocess_wav(in_fpath)
            # - If the wav is already loaded:
            original_wav, sampling_rate = librosa.load(in_fpath)
            preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
            print("Loaded file succesfully")
            embed = encoder.embed_utterance(preprocessed_wav)
            print("Created the embedding")
        
            texts = [args.text]
            texts = g2p(texts)
            print(texts)
            embeds = [embed]
            specs = synthesizer.synthesize_spectrograms(texts, embeds)
            spec = specs[0]
            print("Created the mel spectrogram")
            ## Generating the waveform
            print("Synthesizing the waveform:")
    
            generated_wav = vocoder.infer_waveform(spec)
        
            generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
    
             # if not args.no_sound:
             #     sd.stop()
             #     sd.play(generated_wav, synthesizer.sample_rate)
            
            # Save it on the disk
            fpath = "demo_output_%02d.wav" % num_generated
            print(generated_wav.dtype)
            sf.write(fpath, generated_wav.astype(np.float32), 
                                 synthesizer.sample_rate)
            num_generated += 1
            print("\nSaved output as %s\n\n" % fpath)
    
    
