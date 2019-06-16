# coding: utf-8
import numpy as np
import scipy as sp
import scipy.signal
import scipy.io.wavfile
#import sp.signal
#from scipy import signal
#from scipy.signal import resample_poly

def awgn( signal, snr) :
    """
    Ajoute un bruit blanc gaussien à un signal pour obtenir le SNR cible

    :param signal: Signal brut
    :param snr: Niveau de bruit cible
    """
    puissance_signal_2 = np.linalg.norm( signal ) / np.sqrt( len( signal))

    sigma = puissance_signal_2 / 10**(snr/20)

    bruit = np.random.normal( loc=0, scale=sigma, size=signal.shape)

    return signal + bruit

def spectrogram( signal, fs_in, fs_out, bw=None) :
    """
    Réechantillon un signal de fs_in à fs_out et calcule son spectrogramme

    :param signal: Signal dont on veux calculer le spectrogramme
    :param fs_in: Frequence d'echantilonnage de signal
    :param fs_out: Frequence de réechantilonnage
    :param bw: Si spécifier, retourne un spectrogramme de largeur bw
    """
    signal_resampled = sp.signal.resample_poly( signal, fs_out, fs_in)
   
    nfft = 1024
    (f,t,Sxx) = sp.signal.spectrogram( signal_resampled, fs_out, nperseg=nfft, noverlap=nfft//8, window=sp.signal.get_window(11.1, nfft), return_onesided=False)

    if not bw == None :
        # Trouve l'indice de la première frequence >= à bw/2
        idx = next(i for i,f_i in enumerate( f) if f_i >= bw/2)
        mask = np.arange(1-idx, idx)
        f = f[mask]
        Sxx = Sxx[mask]
    else :
        f = np.fft.fftshift( f)
        Sxx = np.fft.fftshift( Sxx, axes = 0)
    return (f, t, 10*np.log10(Sxx))

def display_spectrogram( f, t, Sxx,protocol,number) :
    import matplotlib.pyplot as plt
    plt.pcolormesh( f, t, Sxx.transpose())
    plt.gca().invert_yaxis()
    #plt.xlabel('Frequency [Hz]')
    #plt.ylabel('Time [sec]')
    #plt.colorbar()
    plt.axis('off')
    plt.savefig('/mnt/disque2/IMT/base/framePNG/'+protocol+'/'+protocol+'_'+number+'.png',bbox_inches = 'tight')
    #plt.show()

if __name__ == "__main__" :
    
    protocol = ['ProtocolA','ProtocolB','ProtocolC','ProtocolD','AIS','DMR','EDACS48','EDACS96','NXDN48','NXDN96']
    N=101 #number of files in foulder + 1 

    for j in range(0,len(protocol)):
        print('processing '+protocol[j])
        for i in range(1,N):
   
            [fe, x] = sp.io.wavfile.read('/mnt/disque2/IMT/base/frame/'+protocol[j]+'/'+protocol[j]+'_'+str(i)+'.wav')
            y = x[:,0] + 1j*x[:,1]
        
            # Ajout d'un bruit blanc gaussien pour obtenir un SNR de 20db
            z = awgn( y, 20)
        
            # Calcul du spectrogramme
            f_out = 50e6 # Le choix de la fréquence de rééchantillonage détermine la résolution spectrale du spectrogramme
            (f,t,Sxx) = spectrogram( z, fe, f_out, bw=1e6)
        
            # Affichage
            display_spectrogram( f, t, Sxx,protocol[j],str(i))
    