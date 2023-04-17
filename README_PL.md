# Stable Diffusion web UI
Interfejs przeglądarkowy oparty na bibliotece Gradio dla Stable Diffusion.

![](screenshot.png)

## Features
[Szczegółowy pokaz funkcji z obrazkami](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features):
- Oryginalne tryby: txt2img i img2img.
- Instalacja jednym kliknięciem i uruchomienie skryptu (ale nadal musisz zainstalować python i git)
- Malowanie na wierzchu
- M0alowanie w środku
- Szkic kolorów
- Prompt Matrix
- Stable Diffusion Zwiększanie rozdzieloczości
- Uwaga, określ części tekstu, na które model powinien zwrócić większą uwagę
    - a man in a `((smoking))` - zwróci większą uwagę na smoking
    - a man in a `(smoking:1.21)` - alternatywna składnia
    - zaznacz tekst i wciśnij `Ctrl+Up` lub `Ctrl+Down` aby automatycznie dostosować uwagę do zaznaczonego tekstu (kod autorstwa anonimowego użytkownika)
- Loopback, wielokrotne uruchamianie przetwarzania img2img
- X/Y/Z plot, sposób na narysowanie trójwymiarowego wykresu obrazów o różnych parametrach
- Inwersja tekstowa
    - posiadanie dowolnej liczby embeddingów i używanie dla nich dowolnych nazw
    - używaj wielu embeddingów z różną liczbą wektorów na token
    - działa z liczbami zmiennoprzecinkowymi o połowie precyzji
    - trenuj embeddingi na 8GB (również raporty o działaniu na 6GB)
- Zakładka Extras z:
    - GFPGAN, sieć neuronowa naprawiająca twarze
    - CodeFormer, narzędzie do przywracania twarzy jako alternatywa dla GFPGAN
    - RealESRGAN, sieć neuronowa upscaler
    - ESRGAN, sieć neuronowa z wieloma modelami innych firm
    - SwinIR i Swin2SR ([zobacz tutaj](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/2092)), skalery sieci neuronowych
    - LDSR, skalowanie superrozdzielczości metodą dyfuzji latentnej
- Opcje zmiany proporcji obrazu
- Wybór metody próbkowania
    - Regulacja wartości eta samplera (mnożnik szumu)
    - Bardziej zaawansowane opcje ustawień szumu
- Przerywanie przetwarzania w dowolnym momencie
- Obsługa kart graficznych o pojemności 4 GB (istnieją również doniesienia o działaniu 2 GB)
- Prawidłowe nasiona dla partii
- Walidacja długości tokena na żywo
- Parametry generowania
     - Parametry używane do generowania obrazów są zapisywane wraz z tym obrazem
     - w kawałkach PNG dla PNG, w EXIF dla JPEG
     - można przeciągnąć obraz do zakładki PNG info, aby przywrócić parametry generowania i automatycznie skopiować je do UI
     - może być wyłączony w ustawieniach
     - przeciągnij i upuść obraz/parametry tekstowe do promptboxa
- Przycisk Read Generation Parameters, ładuje parametry w promptboxie do UI
- Strona ustawień
- Uruchamianie dowolnego kodu Pythona z interfejsu użytkownika (musi być uruchomiony z `--allow-code` aby włączyć)
- Podpowiedzi mouseover dla większości elementów UI
- Możliwość zmiany wartości domyślnych/mix/max/step dla elementów UI poprzez konfigurację tekstową
- Wsparcie dla kafelkowania, pole wyboru do tworzenia obrazów, które mogą być kafelkowane jak tekstury
- Pasek postępu i podgląd generowania obrazu na żywo
    - Możliwość użycia oddzielnej sieci neuronowej do produkcji podglądu z prawie zerowym zapotrzebowaniem na pamięć VRAM lub obliczenia
- Negative prompt, dodatkowe pole tekstowe, które pozwala na wypisanie tego, czego nie chcesz widzieć na generowanym obrazie
- Style, sposób na zapisanie części podpowiedzi i łatwe ich zastosowanie poprzez rozwijanie później
- Wariacje, sposób na wygenerowanie tego samego obrazu, ale z drobnymi różnicami
- Zmiana rozmiaru nasion, sposób na wygenerowanie tego samego obrazu, ale w nieco innej rozdzielczości
- Interrogator CLIP, przycisk, który próbuje odgadnąć podpowiedź z obrazu
- Prompt Editing, sposób na zmianę podpowiedzi w trakcie generowania, powiedzmy, aby zacząć robić arbuza i zmienić na dziewczynę z anime w połowie.
- Batch Processing, przetwarzanie grupy plików za pomocą img2img
- Img2img Alternative, odwrotna metoda Eulera kontroli uwagi krzyżowej
- Highres Fix, wygodna opcja do tworzenia obrazów o wysokiej rozdzielczości jednym kliknięciem bez zwykłych zniekształceń
- Przeładowywanie punktów kontrolnych w locie
- Fuzja punktów kontrolnych, zakładka pozwalająca na połączenie do 3 punktów kontrolnych w jeden
- Skrypty własne](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Custom-Scripts) z wieloma rozszerzeniami od społeczności.
- Composable-Diffusion](https://energy-based-model.github.io/Compositional-Visual-Generation-with-Composable-Diffusion-Models/), sposób na używanie wielu podpowiedzi naraz
     - oddzielanie podpowiedzi za pomocą dużych liter `AND`.
     - obsługuje również wagi dla podpowiedzi: `kot :1.2 ORAZ pies ORAZ pingwin :2.2`.
- Brak limitu tokenów dla podpowiedzi (oryginalna stabilna dyfuzja pozwala na użycie do 75 tokenów)
- Integracja DeepDanbooru, tworzy tagi w stylu danbooru dla podpowiedzi anime
- [xformers](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Xformers), duży wzrost prędkości dla wybranych kart: (dodaj `--xformers` do argumentów linii poleceń)
- poprzez rozszerzenie: [Karta historii](https://github.com/yfszzx/stable-diffusion-webui-images-browser): przeglądaj, kieruj i usuwaj obrazy wygodnie w UI
- Opcja generowania na zawsze
- Zakładka Trening
     - opcje hipersieci i embeddingów
     - Wstępna obróbka obrazów: przycinanie, odbicie lustrzane, autotagging przy użyciu BLIP lub deepdanbooru (dla anime)
- Pomijanie klipów
- Hipersieci
- Loras (to samo co Hypernetworks, ale ładniejsze)
- Sparate UI, gdzie można wybrać, z podglądem, które embeddingi, hypernetworks lub Loras dodać do podpowiedzi
- Możliwość wyboru załadowania innego VAE z ekranu ustawień
- Szacowany czas ukończenia na pasku postępu
- API
- Wsparcie dla dedykowanego [inpainting model](https://github.com/runwayml/stable-diffusion#inpainting-with-stable-diffusion) przez RunwayML
- poprzez rozszerzenie: [Aesthetic Gradients](https://github.com/AUTOMATIC1111/stable-diffusion-webui-aesthetic-gradients), sposób generowania obrazów o określonej estetyce za pomocą embedów clip images (implementacja [https://github.com/vicgalle/stable-diffusion-aesthetic-gradients](https://github.com/vicgalle/stable-diffusion-aesthetic-gradients))
- Wsparcie dla [Stable Diffusion 2.0](https://github.com/Stability-AI/stablediffusion) - zobacz instrukcje w [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#stable-diffusion-20)
- Wsparcie dla [Alt-Diffusion](https://arxiv.org/abs/2211.06679) - zobacz [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#alt-diffusion) po instrukcje
- Teraz bez żadnych złych liter!
- Ładowanie punktów kontrolnych w formacie safetensorów
- Złagodzone ograniczenie rozdzielczości: domeną generowanego obrazu musi być wielokrotność 8, a nie 64
- Teraz z licencją!
- Zmiana kolejności elementów w UI z ekranu ustawień

## Instalacja i uruchomienie
Upewnij się, że wymagane [zależności](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Dependencies) są spełnione i postępuj zgodnie z instrukcjami dostępnymi zarówno dla układów GPU [NVidia](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-NVidia-GPUs) (zalecane), jak i [AMD](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs).

Alternatywnie, skorzystaj z usług online (takich jak Google Colab):

- [Lista usług online](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Online-Services)

### Automatyczna instalacja w systemie Windows
1. Zainstaluj [Python 3.10.6](https://www.python.org/downloads/windows/), zaznaczając opcję "Add Python to PATH".
2. Zainstaluj [git](https://git-scm.com/download/win).
3. Pobierz repozytorium stable-diffusion-webui, na przykład uruchamiając `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`.
4. Uruchomić `webui-user.bat` z Eksploratora Windows jako normalny, nieadministrator, użytkownik.

### Automatyczna instalacja w systemie Linux
1. Zainstaluj zależności:
``bash``
# Debian-based:
sudo apt install wget git python3 python3-venv
# Red Hat-based:
sudo dnf install wget git python3
# Arch-based:
sudo pacman -S wget git python3
```
2. Aby zainstalować w `/home/$(whoami)/stable-diffusion-webui/`, uruchom:
``bash
bash <(wget -qO- https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh)
```
3. Uruchomić `webui.sh`.
### Instalacja na Apple Silicon

Znajdź instrukcje [tutaj](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Installation-on-Apple-Silicon).

## Wkład własny
Oto jak dodać kod do tego repo: [Contributing](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing)

## Dokumentacja
Dokumentacja została przeniesiona z tego README na stronę projektu [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).

## Kredyty
Licencje dla pożyczonego kodu można znaleźć w ekranie `Ustawienia -> Licencje`, a także w pliku `html/licenses.html`.

- Dyfuzja stabilna - https://github.com/CompVis/stable-diffusion, https://github.com/CompVis/taming-transformers
- k-dyfuzja - https://github.com/crowsonkb/k-diffusion.git
- GFPGAN - https://github.com/TencentARC/GFPGAN.git
- CodeFormer - https://github.com/sczhou/CodeFormer
- ESRGAN - https://github.com/xinntao/ESRGAN
- SwinIR - https://github.com/JingyunLiang/SwinIR
- Swin2SR - https://github.com/mv-lab/swin2sr
- LDSR - https://github.com/Hafiidz/latent-diffusion
- MiDaS - https://github.com/isl-org/MiDaS
- Pomysły na optymalizacje - https://github.com/basujindal/stable-diffusion
- Optymalizacja warstwy Cross Attention - Doggettx - https://github.com/Doggettx/stable-diffusion, oryginalny pomysł na edycję promptów.
- Optymalizacja warstwy Cross Attention - InvokeAI, lstein - https://github.com/invoke-ai/InvokeAI (pierwotnie http://github.com/lstein/stable-diffusion)
- Sub-kwadratowa optymalizacja warstwy Cross Attention - Alex Birch (https://github.com/Birch-san/diffusers/pull/1), Amin Rezaei (https://github.com/AminRezaei0x443/memory-efficient-attention)
- Textual Inversion - Rinon Gal - https://github.com/rinongal/textual_inversion (nie używamy jego kodu, ale korzystamy z jego pomysłów).
- Pomysł na upscale SD - https://github.com/jquesnelle/txt2imghd
- Generowanie szumu dla outpaintingu mk2 - https://github.com/parlance-zz/g-diffuser-bot
- Pomysł na interrogator CLIP i pożyczenie trochę kodu - https://github.com/pharmapsychotic/clip-interrogator
- Pomysł na kompozytową dyfuzję - https://github.com/energy-based-model/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch
- xformers - https://github.com/facebookresearch/xformers
- DeepDanbooru - interrogator dla dyfuzorów anime https://github.com/KichangKim/DeepDanbooru
- Próbkowanie w precyzji float32 z float16 UNet - marunine za pomysł, Birch-san za przykładową implementację Diffusers (https://github.com/Birch-san/diffusers-play/tree/92feee6)
- Instrukcja pix2pix - Tim Brooks (gwiazda), Aleksander Hołyński (gwiazda), Aleksiej A. Efros (bez gwiazdy) - https://github.com/timothybrooks/instruct-pix2pix
- Porady dotyczące bezpieczeństwa - RyotaK
- Sampler UniPC - Wenliang Zhao - https://github.com/wl-zhao/UniPC
- Początkowy skrypt Gradio - zamieszczony na 4chan przez użytkownika Anonymous. Dziękuję Anonimowemu użytkownikowi.
- (Ty)