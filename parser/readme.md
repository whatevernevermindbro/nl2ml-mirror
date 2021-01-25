# Сбор и парсинг ноутбуков

### Сбор
Запуск из терминала `python3 ./kaggle.py`

Аргументы (kaggle API):
`--page-size` (default : 1001)
`--language` (default : python)
`--kernel-type` (default : notebook)
`--sort-by` (default : dateCreated)
`--competition`
`--dataset`

Аргументы (фильтры):
`--kaggle_score` (применяется только если выбрано соревнование --competition)
`--minimize_score` (получить значения меньшие или равные пороговому(kaggle_score),
    если не задан - получить значения большие или равные порогу)
`--upvotes`
`--comments`

Передает полученные данные в `notebook_parsing.py` и по умолчанию записывает
результат в файл `../data/code_blocks_new_{date}.csv`.

### Парсинг
Запуск из терминала `python3 ./notebook_parsing.py`

Парсер использует библиотеку BeautifulSoup4, а также tqdm для красивого и удобного прогресбара.
чтобы собрать ссылки на используемые данные, потребовался Selenium. С Selenium используется веб-драйвер Google Chrome.

По умолчанию предполагается, что список ноутбуков, по которым нужно пройтись лежит в файле `../data/kaggle_kernels.csv`. Чтобы выбрать другой путь, нужно изменить константу `KERNELS_PATH` в скрипте `./notebook_parsing.py`.

Также по умолчанию собранные блоки кода сохраняются в файл `../data/code_blocks_new.csv`. Другой путь можно выбрать, изменив константу `CODE_BLOCKS_PATH`.
