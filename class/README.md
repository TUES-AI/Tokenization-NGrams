# Тетрадки от часа

Това са тетрадките от занятието със запазените резултати. За да ги изпълниш, разархивирай целия проект, отвори го в JupyterLab и избери **Restart Kernel and Run All Cells**. Ако нямаш JupyterLab, инсталирай го локално с `python -m pip install jupyterlab`, после от папката на проекта изпълни `python -m jupyterlab`. Използвай `py` или `python3` вместо `python`, ако така стартираш Python. Не мести тетрадките отделно от `data/` и `spike_arcade.html`.

| Тетрадка | Опитай |
|---|---|
| [BPE](tokenization/bpe.ipynb) | В `banana` намери двете двойки `a` + `n`. Проследи защо и двете получават ID 256. Смени броя сливания на 0 и провери дали декодираният текст се променя. |
| [Малък триграмен модел](ngram/ngram.ipynb) | След `the cat` има два пъти `sat` и веднъж `slept`. Коя дума винаги избира greedy? Може ли sampling да избере другата? |
| [5-грамен модел върху Sherlock Holmes](ngram/five_gram.ipynb) | Проследи как моделът скъсява непозната поредица от четири токена. Последната клетка показва стара грешка от часа; изпълни наново целия notebook. |
| [Spike School](ngram/spike_school.ipynb) | Избери **Watch policy → Pause → Step once**. На пистата **Double jump** сравни **Short view** и **Five-tile view**. Какво вижда моделът, преди да избере ход? Контролите работят в Jupyter, не в GitHub. |

### Връзка с домашното

Тетрадките показват идеите на малки примери. Домашното използва отделни текстове на български и английски от Wikipedia, а оценяването няма начални и крайни токени. При оценяване контекстът идва от истинския текст, а при генериране идва от токените, които моделът вече е избрал. Виж [началните стъпки](../README.md) и [точния формат](../docs/FORMAT.md).

### Текстовете за Sherlock Holmes

Arthur Conan Doyle: [The Adventures of Sherlock Holmes](https://www.gutenberg.org/ebooks/1661) и [The Memoirs of Sherlock Holmes](https://www.gutenberg.org/ebooks/834) за обучение, [The Return of Sherlock Holmes](https://www.gutenberg.org/ebooks/108) за проверка. Запазените файлове са преформатирани: абзаците са събрани и приблизително разделени на изречения. Оригиналните издания са достъпни безплатно на посочените връзки.

> This eBook is for the use of anyone anywhere in the United States and most other parts of the world at no cost and with almost no restrictions whatsoever. You may copy it, give it away or re-use it under the terms of the [Project Gutenberg License](notices/PROJECT-GUTENBERG.txt) included with this eBook or online at [www.gutenberg.org](https://www.gutenberg.org/license). If you are not located in the United States, you will have to check the laws of the country where you are located before using this eBook.
