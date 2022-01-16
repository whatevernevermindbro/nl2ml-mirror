# Parser

## Terminology

Kaggle has a lot of references for its objects. Let's clarify the terminology.

* A **link** is a full URL, e.g. `https://www.kaggle.com/data-scientist/cool-solution`
* A **slug** is a human-readable identificator. For most Kaggle objects it is simply the part of the URL after `https://www.kaggle.com/`. The slug of the kernel in previous example is `data-scientist/cool-solution`
* A **numerical ID** is a number that Kaggle uses as an internal reference.

## File descriptions

* [clean_data.py](clean_data.py) goes through every code block in the labeled dataset, removes comments, formats code to conform to the PEP8 standard, removes service labels. This script has 2 CLI arguments:
  * path to the labeled dataset
  * path to the knowledge graph

* [collect_kernels_from_competitions.py](collect_kernels_from_competitions.py) traverses popular competitions that are available through Kaggle API and collects the slugs of the connected kernels.

* [competition_kernels.sh](competition_kernels.sh) collects the kernel slugs for a single competition. This script is used in `collect_kernels_from_competitions.py`. This script has 1 CLI argument which is competition slug.

* [competition_collector.py](competition_collector.py) collects the competition slugs from the Kaggle search engine. This script has 3 CLI arguments:
  * `--competition_count` is the maximum amount of collected competition slugs
  * `--process_id` is the id of the worker
  * `--total_processes` is the total amount of workers

* [competition_tags.py](competition_tags.py) extracts competition tags. The tags include metric as well as metadata like data type or subject. 

* [kernel_collector.py](kernel_collector.py) collects kernel slugs from the Kaggle search engine. This script has 2 CLI arguments: 
  * `--kernel_count` is the maximum amount of collected kernel slugs
  * `--process_id` is the id of the worker. The script assumes that there are 3 workers in total. 

* [kernel_parser.py](kernel_parser.py) takes the kernel slugs from csv file in `KERNEL_FILENAME` variable, then it loads each kernel and splits it into code blocks. This script has 1 CLI argument:
  * `--process_id` is the id of the worker. The script assumes that there are 3 workers in total.

* [unite_kernel_lists.py](unite_kernel_lists.py) takes all csv files from a folder and appends them to the main csv file which is defined in `OLD_LIST` variable.

### Submodules

* `./remote` contains bash scripts for easy access to remote servers. The servers are not in use, but these scripts may be useful if you try to collect more kernels in a distributed fashion.
* `./kaggle_scraping` contains code that does the actual parsing of the pages.

## Known limitations

*This is the state of things during spring'21.*

Kaggle search engine only shows `10 000` objects, even if there are more results.

Kaggle API is limited to `1 100` results per search (no more than `11` pages, `100` results per page max).

## Troubleshooting

* The names of the HTML ids and classes change every time Kaggle is updated. If parser does not work at all, checking the xpaths is worth a shot
