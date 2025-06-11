import os
print("MUSDB_PATH:", os.environ.get("MUSDB_PATH"))
import imageio
import numpy as np
from tqdm import tqdm
import musdb
import dsdtools
import librosa
import random
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MUSDB18', help='Dataset to create spectrograms from')
parser.add_argument('--target', type=str, default='vocals', help='Target to separate (vocals, drums, bass)')
parser.add_argument('--root_dir', type=str, default='/media/bhuvan/E8C7-F0C7/AAC/MUSDB18', help='Path to MUSDB18 dataset')
parser.add_argument('--save_dir', type=str, default='/media/bhuvan/E8C7-F0C7/AAC/', help='Directory to save output data')
opts = parser.parse_args()

def musdb2stft(save_dir='./data/', target='wo_vocals'):
    """
    Create STFT dataset from MUSDB18 or DSD100
    """
    if opts.dataset == 'MUSDB18':
        mus = musdb.DB(is_wav=True)
        tracks = mus.load_mus_tracks(subsets='train')
        print(f"Number of tracks loaded: {len(tracks)}")

    elif opts.dataset == 'DSD100':
        dsd = dsdtools.DB()
        tracks = dsd.load_dsd_tracks(subsets='Dev')
    else:
        sys.exit("Only support DSD100|MUSDB18")

    save_dir = os.path.join(save_dir, target)
    if not os.path.exists(save_dir):
        print("Creating directory: {}".format(save_dir))
        os.makedirs(save_dir)

    for track in tqdm(tracks):
        music = switch_music(target, track)
        music = music.mean(axis=1)
        samplerate = 20480
        music = librosa.resample(y=music, orig_sr=track.rate, target_sr=samplerate)

        fft_size = 512
        hopsamp = fft_size // 8
        part_duration = 255 / (samplerate / hopsamp)
        length_song = music.shape[0]
        batch_size = round(samplerate * part_duration)
        counter = 1

        for shift_duration in tqdm([0, 0.3, 0.6]):
            shift_len = round(shift_duration * samplerate)
            number_of_parts = int(np.floor((length_song - shift_len) / batch_size))
            data = music[shift_len:number_of_parts * batch_size + shift_len]
            data2 = data.reshape(number_of_parts, int(data.size / number_of_parts))
            for row in data2:
                stft_full = librosa.core.stft(row, n_fft=fft_size, hop_length=hopsamp, win_length=fft_size)
                stft_full = stft_full[0:-1, :]
                stft_mag = abs(stft_full)
                stft_mag = np.interp(stft_mag, (stft_mag.min(), stft_mag.max()), (0, 1))
                stft_mag = (stft_mag ** 0.3) * 255
                stft_mag = stft_mag.astype('uint8')
                if opts.dataset == 'MUSDB18':
                    songname = track.name + '_' + str(counter)
                else:
                    songname = track.filename + '_' + str(counter)
                songname = os.path.join(save_dir, songname)
                if stft_mag.sum() != 0:
                    imageio.imwrite(songname + '.png', stft_mag)
                counter += 1

def divide_dataset(source_directory, mixture_directory, dataset_name='vocals_dataset'):
    """
    Create the dataset in a folder format to be used during training
    """
    main_dir = os.path.dirname(source_directory)
    save_dir = os.path.join(main_dir, dataset_name)
    test_names = ['testA', 'testB']
    train_names = ['trainA', 'trainB']
    origins = [source_directory, mixture_directory]

    print("Creating directory: {}".format(save_dir))
    os.makedirs(save_dir, exist_ok=True)
    for i, directory in enumerate(test_names):
        test_dir = os.path.join(save_dir, directory)
        os.makedirs(test_dir, exist_ok=True)
        file_names = os.listdir(origins[i])
        test_files = random.sample(file_names, round(0.05 * len(file_names)))
        for file in test_files:
            os.rename(os.path.join(origins[i], file), os.path.join(test_dir, file))
        os.rename(origins[i], os.path.join(save_dir, train_names[i]))

def switch_music(target, track):
    if target == 'wo_drums':
        music = track.targets['vocals'].audio + track.targets['bass'].audio + track.targets['other'].audio
    elif target == 'wo_vocals':
        music = track.targets['drums'].audio + track.targets['bass'].audio + track.targets['other'].audio
    elif target == 'wo_bass':
        music = track.targets['vocals'].audio + track.targets['drums'].audio + track.targets['other'].audio
    elif target == 'drums':
        music = track.targets['drums'].audio
    elif target == 'bass':
        music = track.targets['bass'].audio
    elif target == 'vocals':
        music = track.targets['vocals'].audio
    elif target == 'mixture':
        music = track.audio * 0.5
    else:
        raise ValueError(f"Unknown target: {target}")
    music = music * 2
    return music

if __name__ == '__main__':
    musdb2stft(save_dir=opts.save_dir, target='mixture')

    if opts.target == 'vocals':
        musdb2stft(save_dir=opts.save_dir, target='wo_vocals')
        divide_dataset(
            source_directory=os.path.join(opts.save_dir, 'wo_vocals'),
            mixture_directory=os.path.join(opts.save_dir, 'mixture'),
            dataset_name='vocals_dataset'
        )
    elif opts.target == 'drums':
        musdb2stft(save_dir=opts.save_dir, target='wo_drums')
        divide_dataset(
            source_directory=os.path.join(opts.save_dir, 'wo_drums'),
            mixture_directory=os.path.join(opts.save_dir, 'mixture'),
            dataset_name='drums_dataset'
        )
    elif opts.target == 'bass':
        musdb2stft(save_dir=opts.save_dir, target='wo_bass')
        divide_dataset(
            source_directory=os.path.join(opts.save_dir, 'wo_bass'),
            mixture_directory=os.path.join(opts.save_dir, 'mixture'),
            dataset_name='bass_dataset'
        )
