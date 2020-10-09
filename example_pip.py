import dpam

loss_fn = dpam.DPAM()
wav_ref = dpam.load_audio('sample_audio/ref.wav')
wav_out = dpam.load_audio('sample_audio/2.wav')

dist = loss_fn.forward(wav_ref,wav_out)
print(dist)
