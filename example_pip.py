import pip_pasm

loss_fn = pip_pasm.PASM()
wav_in = pip_pasm.load_audio('sample_audio/ref.wav')
wav_out = pip_pasm.load_audio('sample_audio/2.wav')

dist = loss_fn.forward(wav_in,wav_out)
print(dist)
