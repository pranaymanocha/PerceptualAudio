import pasm

loss_fn = pasm.PASM()
wav_ref = pasm.load_audio('sample_audio/ref.wav')
wav_out = pasm.load_audio('sample_audio/2.wav')

dist = loss_fn.forward(wav_ref,wav_out)
print(dist)
