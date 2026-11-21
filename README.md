# Tokenization & N-grams

**TUES AI · Домашно**

Направи три токенизатора за един и същ корпус от българска и английска Wikipedia. Сравни колко място заемат токените и колко добре модел предсказва следващия токен. После обучи свой n-грамов модел и го сравни с референтния.

[Предаване и класации](https://tuesai.org/homework/tokenization-ngrams) · [Формат на JSON](docs/FORMAT.md) · [Тетрадки от часа](class/README.md) · [Упражнение: генериране](docs/SAMPLING.md)

## 1. Пусни базовия вариант

Клонирай [хранилището](https://github.com/TUES-AI/Tokenization-NGrams) или избери **Code → Download ZIP** и разархивирай. Отвори терминал в папката с този README. Трябват ти Python 3.12+ и обикновен лаптоп. Няма пакети за инсталиране, GPU или допълнителни файлове за сваляне. Ако стартираш Python с `python3` или `py`, използвай тази команда вместо `python`.

```sh
python -m starter.build byte --out output/byte.json
python -m starter.evaluate --examples output/byte.json
```

Първата команда създава готов за качване byte JSON. Втората го сравнява с два малки **необучени** примера за word и BPE върху 8 откъса от всеки език за всеки дял. Таблицата показва `overall`, `bg` и `en`; `-` означава, че няма стойност. Примерите са само за разчитане на таблицата.

### Зареди корпуса

| Дял | Файл | Откъси общо | На език |
|---|---|---:|---:|
| Обучение | [`train.jsonl.gz`](data/train.jsonl.gz) | 16 380 | 8 190 |
| Проверка | [`validation.jsonl.gz`](data/validation.jsonl.gz) | 3 060 | 1 530 |

Всеки ред е UTF-8 JSON с две полета: `lang` (`bg` или `en`) и `text` (откъсът). Българските и английските редове се редуват. От главната папка можеш да прочетеш първите два реда направо от архива:

```python
import gzip
import json
from itertools import islice

with gzip.open("data/train.jsonl.gz", "rt", encoding="utf-8") as source:
    for line in islice(source, 2):
        row = json.loads(line)
        print(row["lang"], row["text"])
```

За да заредиш и двата дяла в паметта, групирани по език:

```python
from data.loader import load_split

train = load_split("train")           # {"bg": [...], "en": [...]}
validation = load_split("validation")
```

Обучавай само върху `train`. Ползвай `validation` за сравнение и избор на настройки. Сайтът оценява върху отделен набор. Виж [описанието и лиценза на корпуса](data/README.md).

## 2. Сравни и предай три токенизатора

| Вариант | Твоята задача |
|---|---|
| **Byte** | Използвай базовия вариант: един ID за всеки UTF-8 байт. Провери на ръка с `AБ`. |
| **Word** | Довърши `train_word` в [`starter/build.py`](starter/build.py). Започни с всички различни думи от избраните `train` откъси на двата езика, разделени по празни знаци. Една и съща дума има един ID в общия речник. Ако намалиш `--size`, остави най-честите думи и сравни какво губиш. Непознатите думи се кодират с оригиналните байтове. За допълнителни разделители виж `boundaries` във [формата](docs/FORMAT.md#words-with-byte-fallback). |
| **BPE** | Довърши `train_bpe`. Започни с 256 ID за байтовете (0–255) и учи по едно сливане на съседна двойка. Записвай сливанията в научения ред; максимум 65 280 сливания (65 536 ID общо, по 16 бита на токен). |

```sh
python -m starter.build word --rows 32 --size 4096 --out output/word.json
python -m starter.build bpe --rows 32 --size 128 --out output/bpe.json
python -m starter.evaluate --rows 32 output/byte.json output/word.json output/bpe.json
```

`--rows` на `build` задава броя обучаващи откъси **на език**, а `--size` ограничава думите или сливанията. Първо пробвай с малки стойности. Сравнявай вариантите върху еднакви редове: `--rows` на `evaluate` увеличава локалната проверка до [лимитите на оценителя](docs/FORMAT.md#limits-and-local-api).

По-малък `token_count` не гарантира по-малък `total_bits`: по-широките ID и самият речник заемат място. По-висока точност също не означава по-нисък **BPB** (битове на оригинален байт); точността брои правилните избори, а BPB отчита вероятността, дадена на всеки верен токен. Сравнявай BPB между варианти без загуба на текст.

Влез в сайта с училищния си акаунт и качи `byte.json`, `word.json` и `bpe.json` поотделно. Качвай само JSON, без код, notebook или ZIP. **Класациите във всички категории са публични. До резултата се вижда показваното име от училищния ти акаунт.** Варианти със загуба на информация получават обратна връзка, но не участват в класациите.

## 3. Обучи и предай n-грами

```mermaid
flowchart LR
    A["train: BG + EN"] --> B["Общ токенизатор"]
    B --> C["Броячи по език"]
    B --> D["Токени във validation"]
    V["validation: BG + EN"] --> D
    C --> E["Точност и BPB"]
    D --> E
    classDef data fill:#e1f2fd,stroke:#297298,color:#123444;
    classDef model fill:#eee8fa,stroke:#785bb2,color:#2c2042;
    classDef metric fill:#e1f5e8,stroke:#3f8050,color:#163522;
    class A,V,D data;
    class B,C model;
    class E metric;
```

Довърши `train_ngrams` и го пусни за **всеки от трите токенизатора**. Подадената функция `encode(document)` връща точно ID-тата, които ползва оценителят. Брой съседните последователности от 1 до 3, отделно за BG и EN, без да пресичаш граница между откъси. Виж [формата и лимитите за броячите](docs/FORMAT.md#optional-submitted-n-gram-counts).

```sh
python -m starter.build byte --rows 32 --order 3 --tokenizer-from output/byte.json --out output/byte-ngram.json
python -m starter.build word --rows 32 --order 3 --tokenizer-from output/word.json --out output/word-ngram.json
python -m starter.build bpe --rows 32 --order 3 --tokenizer-from output/bpe.json --out output/bpe-ngram.json
python -m starter.evaluate --rows 32 output/byte-ngram.json output/word-ngram.json output/bpe-ngram.json
```

`--tokenizer-from` запазва същите ID от първия JSON. Качи трите нови `*-ngram.json` файла поотделно. Локално референтният модел също брои до ред 3 върху същите 32 обучаващи откъса на език. Ако броячите ти са верни, `submitted_accuracy` и `submitted_bpb` трябва да съвпадат с `reference_accuracy` и `reference_bpb` за всеки от трите токенизатора. При разлика първо провери дали броиш и по-късите n-грами и дали спираш на границата на всеки откъс.

Попълни от редовете `overall` на локалната проверка. Сравни и редовете `bg` и `en` за една разлика между двата езика.

| Модел | Токени / общо битове | Реф. точност / BPB | Твои точност / BPB |
|---|---|---|---|
| Byte | … / … | … / … | … / … |
| Word | … / … | … / … | … / … |
| BPE | … / … | … / … | … / … |

**Втори опит:** За един от токенизаторите промени само реда от 3 на 2. Например за byte:

```sh
python -m starter.build byte --rows 32 --order 2 --tokenizer-from output/byte.json --out output/byte-bigram.json
python -m starter.evaluate --rows 32 output/byte-ngram.json output/byte-bigram.json
```

Запиши как се променят точността и BPB, когато моделът вижда най-много един предишен токен вместо два. Ако някоя метрика остане същата, предложи обяснение. Този опит остава в локалния отчет.

## 4. Генерирай и обясни резултатите

Направи [упражнението с greedy и претеглено теглене](docs/SAMPLING.md). Запази кратък отчет за себе си с няколко извадки, метриките и какво според теб обяснява разликите.

### Видеа от Andrej Karpathy

- [Let’s build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) · байтове, сливане на двойки и BPE
- [The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo) · броене, вероятности и генериране с биграмен модел

[Условия за ползване на учебните материали](USE.md)
