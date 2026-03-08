# YOLOv8 Office Object Detection

Bu proje, ofis ortamında sık görülen nesneleri tespit etmek için **Ultralytics YOLOv8** kullanır. Veri kümesi üzerinden model eğitimi yapılabilir, eğitilmiş ağırlıklarla ya da varsayılan YOLOv8 modeliyle tek bir görsel üzerinde tahmin alınabilir.

Projede tespit edilen sınıflar:

- `klavye`
- `monitor`
- `mouse`
- `sandalye`

## Projenin amacı

Bu çalışmanın amacı, küçük bir özel veri kümesi ile YOLOv8 tabanlı nesne tespiti akışını uçtan uca göstermektir:

- veri kümesini tanımlama,
- modeli eğitme,
- eğitilen ağırlıkları bulma,
- örnek bir görsel üzerinde tahmin çalıştırma,
- çıktı dosyalarını kaydetme.

## Proje yapısı

```text
.
├── Dataset/
│   ├── data.yaml
│   └── train/images/
├── runs/
├── main.py
├── predict.py
├── requirements.txt
└── yolov8n.pt
```

Kısa açıklama:

- `Dataset/data.yaml`: eğitim verisinin yolu, sınıf sayısı ve sınıf isimleri.
- `main.py`: projenin ana komut satırı arayüzü; eğitim ve tahmin işlemleri burada yönetilir.
- `predict.py`: sadece `main.py predict` komutunu çağıran küçük bir yardımcı dosya.
- `yolov8n.pt`: eğitim başlatılırken kullanılan temel YOLOv8 model ağırlığı.
- `runs/detect/`: eğitim ve tahmin çıktılarının yazıldığı klasör.

## Kurulum

Önce bağımlılıkları yükleyin:

```bash
python -m pip install -r requirements.txt
```

## Kullanım

### Eğitim

Modeli eğitmek için:

```bash
python main.py train --epochs 15 --imgsz 640
```

Eğer sadece şu komutu çalıştırırsanız:

```bash
python main.py
```

program bunu varsayılan olarak eğitim komutu gibi yorumlar ve şu değerleri kullanır:

- `epochs = 15`
- `imgsz = 640`

### Tahmin

Varsayılan ayarlarla tahmin almak için:

```bash
python main.py predict
```

Belirli bir görsel üzerinde tahmin almak için:

```bash
python main.py predict --source Dataset/train/images/example.jpg
```

Belirli bir ağırlık dosyasıyla tahmin almak için:

```bash
python main.py predict --source Dataset/train/images/example.jpg --weights runs/detect/train/weights/best.pt
```

Sonucu ekranda göstermek için:

```bash
python main.py predict --show
```

Sonuç dosyalarını kaydetmeden çalıştırmak için:

```bash
python main.py predict --no-save
```

İsterseniz kısa yol olarak şu dosyayı da kullanabilirsiniz:

```bash
python predict.py
```

## `main.py` dosyası ne yapıyor?

Bu projenin asıl mantığı `main.py` içinde bulunur. Dosya, hem eğitim hem de tahmin sürecini tek yerden yönetir.

### 1. Proje yollarını tanımlıyor

Dosyanın başında proje klasörü, veri kümesi dosyası, temel model ve çıktı klasörleri tanımlanır. Böylece kod, bulunduğu dizine göre doğru dosyalara erişebilir.

Tanımlanan ana yollar:

- proje kökü,
- `Dataset/` klasörü,
- `Dataset/data.yaml`,
- `yolov8n.pt`,
- `runs/detect/`.

Bu yapı sayesinde kod içinde sabit ve dağınık dosya yolları kullanmak yerine merkezi bir yol yönetimi sağlanır.

### 2. Gerekli dosyaların varlığını kontrol ediyor

`ensure_exists()` fonksiyonu, verilen dosya veya klasörün gerçekten var olup olmadığını kontrol eder. Eğer dosya yoksa anlamlı bir hata üretir.

Bu kontrol özellikle şu durumlarda önemlidir:

- `yolov8n.pt` eksikse eğitim başlayamaz,
- `Dataset/data.yaml` eksikse veri kümesi okunamaz,
- tahmin için seçilen görsel yoksa inference yapılamaz.

Yani bu fonksiyon, programın sessizce bozulması yerine kullanıcıya net hata vermesini sağlar.

### 3. Varsayılan örnek görseli buluyor

`find_default_image()` fonksiyonu, kullanıcı `--source` vermediğinde `Dataset/train/images/` klasöründe örnek bir görsel arar.

Fonksiyon şu uzantıları sırayla kontrol eder:

- `.jpg`
- `.jpeg`
- `.png`
- `.bmp`

İlk bulunan görsel tahmin için kullanılır. Böylece kullanıcı ekstra yol vermeden doğrudan `python main.py predict` komutunu çalıştırabilir.

### 4. En son eğitilmiş ağırlıkları buluyor

`find_latest_trained_weights()` fonksiyonu, `runs/detect/` altında oluşan eğitim klasörlerini tarar ve en son oluşturulmuş `best.pt` dosyasını bulur.

Bu önemli bir kolaylık sağlar çünkü kullanıcı her seferinde manuel olarak ağırlık dosyasının yolunu yazmak zorunda kalmaz.

Tahmin sırasında ağırlık seçme sırası şöyledir:

1. Kullanıcı `--weights` verdiyse önce o kullanılır.
2. Verilmediyse en son eğitilen `best.pt` aranır.
3. O da yoksa `yolov8n.pt` ile devam edilir.

Bu yaklaşım, projeyi hem yeni başlayan biri için kolaylaştırır hem de eğitim sonrası otomatik kullanım sağlar.

### 5. Modeli eğitiyor

`train_model(epochs, imgsz)` fonksiyonu eğitimden sorumludur.

Fonksiyonun yaptığı işler:

- temel model dosyasının varlığını kontrol eder,
- `data.yaml` dosyasının varlığını kontrol eder,
- `YOLO(str(DEFAULT_MODEL))` ile modeli yükler,
- `model.train(...)` çağrısıyla eğitimi başlatır.

Eğitim sırasında kullanılan önemli parametreler:

- `data=str(DATA_YAML)`: veri kümesi tanımı,
- `epochs=epochs`: eğitim tekrar sayısı,
- `imgsz=imgsz`: giriş görüntü boyutu,
- `device='cpu'`: eğitim CPU üzerinde çalışır,
- `workers=0`: veri yükleme işçi sayısı sıfırdır.

Eğitim bittiğinde çıktı klasörü ekrana yazdırılır. Böylece kullanıcı model ağırlıklarının nereye kaydedildiğini kolayca görebilir.

### 6. Görsel üzerinde tahmin yapıyor

`predict_image(source, weights, save, show)` fonksiyonu tek bir görsel üzerinde tahmin çalıştırır.

İşleyiş sırası şöyledir:

- kullanıcı kaynak görsel vermediyse varsayılan örnek görsel seçilir,
- seçilen görselin varlığı kontrol edilir,
- kullanılacak ağırlık dosyası belirlenir,
- model yüklenir,
- `model.predict(...)` ile tahmin yapılır.

Buradaki önemli parametreler:

- `source=str(source)`: tahmin yapılacak görsel,
- `save=save`: çıktı dosyalarını kaydetme tercihi,
- `show=show`: sonucu pencere olarak gösterme tercihi,
- `device='cpu'`: tahmin CPU üzerinde çalışır.

İşlem sonunda hangi modelin kullanıldığı yazdırılır. Eğer kayıt açıksa, çıktıların kaydedildiği klasör de ayrıca gösterilir.

### 7. Komut satırı arayüzü kuruyor

`build_parser()` fonksiyonu `argparse` kullanarak iki alt komut tanımlar:

- `train`
- `predict`

`train` komutunun parametreleri:

- `--epochs`
- `--imgsz`

`predict` komutunun parametreleri:

- `--source`
- `--weights`
- `--show`
- `--no-save`

Bu sayede proje terminalden daha düzenli ve kontrollü şekilde kullanılabilir.

### 8. Program akışını yönetiyor

`main()` fonksiyonu kullanıcıdan gelen komutları çözer ve uygun fonksiyonu çağırır.

Akış şu şekildedir:

- argümanlar okunur,
- komut verilmemişse varsayılan olarak `train` seçilir,
- `train` ise `train_model(...)` çağrılır,
- `predict` ise `predict_image(...)` çağrılır,
- hata oluşursa ekrana yazdırılır ve program `1` koduyla kapanır.

Bu yapı sayesinde kod doğrudan dosya açılır açılmaz çalışmaz; sadece `if __name__ == '__main__':` bloğu içinde güvenli şekilde başlatılır.

## Veri kümesi hakkında not

`Dataset/data.yaml` dosyasında şu bilgiler yer alır:

- veri kümesi kök yolu: `Dataset`
- eğitim görselleri: `train/images`
- doğrulama görselleri: `train/images`
- sınıf sayısı: `4`

Bu projede doğrulama için de aynı görsel klasörü kullanılmış. Küçük ders/proje çalışmalarında bu kabul edilebilir olsa da gerçek projelerde eğitim ve doğrulama verisini ayrı klasörlerde tutmak daha sağlıklıdır.

## Düzeltilen noktalar

Bu repo içinde önceki sürüme göre şu problemler giderildi:

- eksik `ultralytics` bağımlılığı eklendi,
- `Dataset/data.yaml` içindeki yol daha doğru hale getirildi,
- `predict.py` doğrudan ana tahmin akışını çağıracak şekilde sadeleştirildi,
- `main.py` daha güvenli bir komut satırı akışına dönüştürüldü,
- tahmin tarafına otomatik örnek görsel ve otomatik ağırlık seçimi eklendi.

## Özet

Bu proje küçük ama düzenli bir YOLOv8 uygulamasıdır. En önemli nokta, `main.py` dosyasının sadece modeli çalıştırmakla kalmayıp aynı zamanda:

- dosya kontrollerini yapması,
- eğitim ve tahmini ayrı fonksiyonlara bölmesi,
- en uygun ağırlığı otomatik seçmesi,
- komut satırı parametreleriyle esnek kullanım sunmasıdır.

Yani `main.py`, projenin hem kontrol merkezi hem de kullanıcı arayüzüdür.
